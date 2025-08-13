#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import queue
import signal
import random
import logging
import logging.handlers
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
import psutil
from multiprocessing import Process, Queue, set_start_method

# 为了兼容不同平台的多进程
try:
    set_start_method("spawn")
except RuntimeError:
    pass

# =============================
# 配置 & 常量
# =============================

TOKEN_PATH = os.path.abspath("./.token")
LOG_DIR = os.path.abspath("./log")

EXIT_SIGNALS = (signal.SIGINT, signal.SIGTERM)

DEFAULT_CONFIG = {
    "server_url": "http://127.0.0.1:8000",
    "key_path": "example",
    "client_group": "Group-1",
    "client_name": "Client-1",

    "heartbeat_period": 5,
    "max_jitter": 1.5,
    "sample_timeout": 2.5,     # 每项采样的超时时间（秒）

    "retry": {
        "max_attempts": 3,
        "backoff_base": 1.0,   # 秒
        "backoff_cap": 10.0    # 秒
    },

    "features": {
        "cpu": True,
        "mem": True,
        "gpu": True
    },

    "gpu": {
        "indices": [],  # 空=全部
        "backend_preference": ["pynvml", "nvidia-smi"],
        "nvidia_smi_path": "nvidia-smi"
    },

    "tls": {
        "verify": True,
        "ca_path": ""
    },

    "log_level": "info"
}

# =============================
# 日志
# =============================

def ensure_log_dir_and_rotate(retain_days: int = 7) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    # 删除过期日志
    now = dt.datetime.now()
    for fn in os.listdir(LOG_DIR):
        if not fn.endswith(".log"):
            continue
        try:
            date_str = fn[:-4]  # 去掉 .log
            file_date = dt.datetime.strptime(date_str, "%Y-%m-%d")
            delta = now.date() - file_date.date()
            if delta.days > retain_days:
                os.remove(os.path.join(LOG_DIR, fn))
        except Exception:
            pass
    today = now.strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"{today}.log")

def setup_logger(level: str = "info") -> logging.Logger:
    loglevel = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }.get(level.lower(), logging.INFO)

    logfile = ensure_log_dir_and_rotate(retain_days=7)

    logger = logging.getLogger("kt-client")
    logger.setLevel(loglevel)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(loglevel)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    return logger

# =============================
# 工具
# =============================

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def bytes_to_gb(x: float) -> float:
    return float(x) / (1024.0 ** 3)

def clamp01(x: float) -> float:
    return 0.0 if x is None else max(0.0, min(1.0, float(x)))

def build_url(base: str, key_path: str, tail: str) -> str:
    base = base.rstrip("/")
    key_path = key_path.rstrip("/")
    tail = tail.rstrip("/")
    return f"{base}/{key_path}/{tail}"

def requests_verify_opts(tls_cfg: dict) -> Any:
    verify = tls_cfg.get("verify", True)
    ca_path = tls_cfg.get("ca_path", "")
    if verify and ca_path:
        return ca_path
    return bool(verify)

# =============================
# 子进程采集（带超时）
# =============================

def _collect_cpu(interval: float, out_q: Queue):
    try:
        arr = psutil.cpu_percent(interval=interval, percpu=True)
        res = [float(v) / 100.0 for v in arr]  # 0~1
        out_q.put(res)
    except Exception:
        out_q.put(None)

def _collect_mem(out_q: Queue):
    try:
        vm = psutil.virtual_memory()
        used_gb = bytes_to_gb(vm.total - vm.available)
        total_gb = bytes_to_gb(vm.total)
        out_q.put([used_gb, total_gb])
    except Exception:
        out_q.put(None)

def _collect_gpu_pynvml(indices: List[int], out_q: Queue):
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        selected = indices if indices else list(range(device_count))
        res = []
        for i in selected:
            if i < 0 or i >= device_count:
                continue
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(h)
            used_gb = bytes_to_gb(meminfo.used)
            total_gb = bytes_to_gb(meminfo.total)
            res.append({"usage": clamp01(util / 100.0), "mem": [used_gb, total_gb]})
        pynvml.nvmlShutdown()
        out_q.put(res)
    except Exception:
        out_q.put(None)

def _collect_gpu_nvsmi(indices: List[int], nvsmi_path: str, out_q: Queue):
    import subprocess
    try:
        fields = "utilization.gpu,memory.used,memory.total,index"
        cmd = [nvsmi_path, f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
        if proc.returncode != 0:
            out_q.put(None)
            return
        lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
        rows = []
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) != 4:
                continue
            util = float(parts[0])  # %
            used_mb = float(parts[1])
            total_mb = float(parts[2])
            idx = int(parts[3])
            rows.append((idx, util, used_mb, total_mb))
        rows.sort(key=lambda x: x[0])
        selected_idx = set(indices) if indices else None
        res = []
        for idx, util, used_mb, total_mb in rows:
            if selected_idx is not None and idx not in selected_idx:
                continue
            used_gb = used_mb / 1024.0
            total_gb = total_mb / 1024.0
            res.append({"usage": clamp01(util / 100.0), "mem": [used_gb, total_gb]})
        out_q.put(res)
    except Exception:
        out_q.put(None)

def run_with_timeout(target, args: tuple, timeout: float):
    """子进程执行 target(*args)，超时则终止并返回 None"""
    q = Queue(maxsize=1)
    p = Process(target=target, args=(*args, q), daemon=True)
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        p.join(0.2)
        return None
    try:
        return q.get_nowait()
    except queue.Empty:
        return None

# =============================
# 采集调度
# =============================

def collect_metrics(cfg: dict, logger: logging.Logger) -> Dict[str, Any]:
    feats = cfg.get("features", {})
    gpu_cfg = cfg.get("gpu", {})
    hb_period = float(cfg.get("heartbeat_period", 5.0))
    sample_timeout = float(cfg.get("sample_timeout", 2.5))

    interval = min(max(hb_period, 0.1), 1.0)  # 0.1~1.0
    payload: Dict[str, Any] = {}

    # CPU
    if feats.get("cpu", True):
        cpu = run_with_timeout(_collect_cpu, (interval,), sample_timeout)
        if cpu is not None:
            payload["cpu"] = cpu

    # MEM
    if feats.get("mem", True):
        mem = run_with_timeout(_collect_mem, tuple(), sample_timeout)
        if mem is not None:
            payload["mem"] = mem

    # GPU
    if feats.get("gpu", True):
        indices = gpu_cfg.get("indices", [])
        backend_pref = gpu_cfg.get("backend_preference", ["pynvml", "nvidia-smi"])
        nvsmi_path = gpu_cfg.get("nvidia_smi_path", "nvidia-smi")

        gpu = None
        for b in backend_pref:
            if b == "pynvml":
                gpu = run_with_timeout(_collect_gpu_pynvml, (indices,), sample_timeout)
            elif b == "nvidia-smi":
                gpu = run_with_timeout(_collect_gpu_nvsmi, (indices, nvsmi_path), sample_timeout)
            if gpu is not None:
                break
        if gpu is not None:
            payload["gpu"] = gpu

    return payload

# =============================
# 注册 & 心跳
# =============================

def read_token() -> Optional[str]:
    if os.path.exists(TOKEN_PATH):
        try:
            with open(TOKEN_PATH, "r", encoding="utf-8") as f:
                tok = f.read().strip()
                return tok or None
        except Exception:
            return None
    return None

def write_token(token: str):
    save_text(TOKEN_PATH, token)

def register(cfg: dict, logger: logging.Logger) -> str:
    url = build_url(cfg["server_url"], cfg["key_path"], "register-client")
    payload = {
        "client_group": cfg["client_group"],
        "client_name": cfg["client_name"],
        "client_config": {
            "heartbeat_period": cfg.get("heartbeat_period", 5)
        }
    }
    verify_opt = requests_verify_opts(cfg.get("tls", {}))
    logger.info(f"注册中：POST {url}")
    r = requests.post(url, json=payload, timeout=5, verify=verify_opt)
    j = r.json()
    if r.status_code != 200 or j.get("status") != "ok":
        raise RuntimeError(f"注册失败：HTTP {r.status_code} {j}")
    token = j["token"]
    write_token(token)
    logger.info("注册成功，token 已保存到 ./.token")
    return token

def heartbeat(cfg: dict, token: str, payload: dict, logger: logging.Logger) -> Tuple[bool, str]:
    url = build_url(cfg["server_url"], cfg["key_path"], "heart-beat")
    verify_opt = requests_verify_opts(cfg.get("tls", {}))
    data = {"client_token": token, "client_info": payload}
    r = requests.post(url, json=data, timeout=5, verify=verify_opt)
    try:
        j = r.json()
    except Exception:
        j = {}
    # 你的服务端：HTTP 400 且 message 含 "invalid" -> token 无效
    if r.status_code == 400 and "invalid" in (j.get("message", "").lower()):
        return False, "invalid-token"
    if r.status_code != 200 or j.get("status") != "ok":
        return False, f"http-{r.status_code}"
    return True, "ok"

# =============================
# 运行循环
# =============================

_RUNNING = True
def _signal_handler(signum, frame):
    global _RUNNING
    _RUNNING = False

for sig in EXIT_SIGNALS:
    signal.signal(sig, _signal_handler)

def main(config_path: str):
    # 读配置
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        cfg.update(load_json(config_path))
    logger = setup_logger(cfg.get("log_level", "info"))

    logger.info("KT Client 启动")
    logger.info(f"配置文件：{os.path.abspath(config_path)}")
    logger.info(f"server_url={cfg['server_url']} key_path={cfg['key_path']} group={cfg['client_group']} name={cfg['client_name']}")

    # 取 token（无则注册）
    token = read_token()
    if not token:
        try:
            token = register(cfg, logger)
        except Exception as e:
            logger.error(f"注册失败：{e}")
            sys.exit(2)
    else:
        # —— 启动时“探测心跳”验证 token ——（若 invalid 则自动重新注册）
        try:
            ok, reason = heartbeat(cfg, token, {}, logger)  # 空 payload 用于探测
            if not ok and reason == "invalid-token":
                logger.warning("本地 token 失效（启动探测），自动重新注册...")
                token = register(cfg, logger)
            elif not ok:
                logger.warning(f"启动探测心跳失败（{reason}），暂继续使用现有 token。")
        except requests.exceptions.RequestException as re:
            logger.warning(f"启动时无法连接服务端：{re}；进入重试流程。")

    # 心跳循环
    hb_period = float(cfg.get("heartbeat_period", 5.0))
    jitter = float(cfg.get("max_jitter", 0.0))
    retry_cfg = cfg.get("retry", {})
    max_attempts = int(retry_cfg.get("max_attempts", 3))
    backoff_base = float(retry_cfg.get("backoff_base", 1.0))
    backoff_cap = float(retry_cfg.get("backoff_cap", 10.0))

    while _RUNNING:
        t0 = time.time()
        try:
            payload = collect_metrics(cfg, logger)
        except Exception as e:
            logger.warning(f"采集异常：{e}")
            payload = {}

        # 发心跳（带重试）
        ok = False
        invalid_token = False
        for attempt in range(max_attempts):
            try:
                ok, reason = heartbeat(cfg, token, payload, logger)
                if ok:
                    logger.debug(f"心跳成功：{reason}")
                    break
                else:
                    if reason == "invalid-token":
                        logger.error("运行中收到 invalid token，客户端按要求退出。")
                        invalid_token = True
                        break
                    else:
                        logger.warning(f"心跳失败（{reason}），准备重试 {attempt+1}/{max_attempts}")
            except requests.exceptions.RequestException as re:
                logger.warning(f"心跳网络异常：{re}，准备重试 {attempt+1}/{max_attempts}")

            # 退避
            backoff = min(backoff_base * (2 ** attempt), backoff_cap)
            time.sleep(backoff)

        if invalid_token:
            sys.exit(3)

        # 休眠到下一轮（含抖动）
        t_spent = time.time() - t0
        base_sleep = max(0.0, hb_period - t_spent)
        time.sleep(base_sleep)
        if jitter > 0:
            time.sleep(random.uniform(0, jitter))

    logger.info("收到退出信号，已退出。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python client.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])
