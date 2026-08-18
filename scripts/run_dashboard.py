import subprocess
import sys

from app.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "dashboard/streamlit_app.py",
            f"--server.address={settings.dashboard_host}",
            f"--server.port={settings.dashboard_port}",
            "--server.headless=true",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
