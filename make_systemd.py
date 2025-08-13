#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import shutil

DEFAULT_SERVICE_NAME = "ktboard-client"

def main():
    # 工作目录：当前目录
    work_dir = Path.cwd().resolve()
    python_path = Path(sys.executable).resolve()
    client_py = work_dir / "app.py"
    # 配置文件：可传参覆盖，默认 ./client_config.json
    config_path = (Path(sys.argv[1]).resolve() if len(sys.argv) >= 2
                   else (work_dir / "config.json").resolve())
    service_name = (sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_SERVICE_NAME)

    # 基本检查
    if not client_py.exists():
        print(f"[ERROR] 找不到 {client_py}")
        sys.exit(1)
    if not config_path.exists():
        print(f"[ERROR] 找不到配置文件：{config_path}\n"
              f"用法：python {Path(__file__).name} <config.json> [service_name]")
        sys.exit(1)

    # 日志目录：./log  —— 自动创建
    log_dir = work_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 用户级 systemd unit 路径
    user_systemd_dir = Path.home() / ".config" / "systemd" / "user"
    user_systemd_dir.mkdir(parents=True, exist_ok=True)
    service_file = user_systemd_dir / f"{service_name}.service"

    # 拼接 unit 内容
    unit = f"""[Unit]
Description=KT Board Client (user service)
After=network-online.target

[Service]
Type=simple
ExecStart={python_path} {client_py} {config_path}
WorkingDirectory={work_dir}
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=30
# 将 stdout/stderr 也写入本地文件（你的 client.py 也会写日期日志；这两行是额外兜底）
StandardOutput=append:{(log_dir / f"{service_name}.out.log")}
StandardError=append:{(log_dir / f"{service_name}.err.log")}

[Install]
WantedBy=default.target
"""

    # 写入 unit 文件
    try:
        service_file.write_text(unit, encoding="utf-8")
        print(f"[OK] 写入用户级 unit: {service_file}")
    except Exception as e:
        print(f"[ERROR] 无法写入 {service_file}: {e}")
        sys.exit(1)

    # 检查 systemctl 是否存在
    if shutil.which("systemctl") is None:
        print("[ERROR] 未找到 systemctl。请在支持 systemd 的 Linux 上运行。")
        sys.exit(1)

    # 执行 --user 管理命令
    def run_cmd(cmd):
        print("[CMD]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    try:
        run_cmd(["systemctl", "--user", "daemon-reload"])
        run_cmd(["systemctl", "--user", "enable", "--now", f"{service_name}.service"])
        print(f"[OK] 已启动并设置登录自启（用户级）：{service_name}.service")
        print(f"[INFO] 日志目录：{log_dir} （已自动创建）")
    except subprocess.CalledProcessError as e:
        # 常见情况：Failed to connect to bus（用户会话总线没起来）
        print("\n[ERROR] 无法执行 systemctl --user 命令。可能原因：")
        print("  - 当前会话没有用户级 systemd（例如在某些容器/WSL/非图形会话中）")
        print("  - 或需要启用 linger 以便在未登录时也能拉起：")
        print("      loginctl enable-linger $USER")
        print("  再次登录后重试：")
        print(f"      systemctl --user daemon-reload && systemctl --user enable --now {service_name}.service")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
