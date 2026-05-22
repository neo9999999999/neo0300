"""
체이닝 러너: walk_forward 종료 대기 → overnight_optimizer 자동 시작.
"""
import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path


def is_running(name: str) -> bool:
    """프로세스 이름이 실행 중인지 확인."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", name],
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def main():
    print(f"===== 체이닝 시작: {datetime.now()} =====", flush=True)

    # 1) walk_forward 종료 대기
    print("[대기] run_walk_forward.py 종료 기다리는 중...", flush=True)
    waited = 0
    while is_running("run_walk_forward.py"):
        time.sleep(60)
        waited += 60
        if waited % 600 == 0:  # 10분마다
            print(f"[대기 중] {waited//60}분 경과...", flush=True)

    print(f"[OK] walk_forward 종료 확인: {datetime.now()}", flush=True)
    time.sleep(5)  # 캐시 flush 여유

    # 2) Overnight Optimizer 실행
    print(f"\n[시작] Overnight Optimizer: {datetime.now()}", flush=True)
    os.environ["N_RANDOM"] = "1500"
    os.environ["N_REFINE"] = "1000"

    here = Path(__file__).parent
    py = here / "venv" / "bin" / "python3"
    script = here / "overnight_optimizer.py"

    proc = subprocess.run(
        [str(py), "-W", "ignore", "-u", str(script)],
        cwd=str(here),
        env={**os.environ},
    )
    print(f"\n===== 체이닝 완료: {datetime.now()} (exit {proc.returncode}) =====",
          flush=True)


if __name__ == "__main__":
    main()
