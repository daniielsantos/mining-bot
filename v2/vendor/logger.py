from __future__ import annotations

from datetime import datetime
from pathlib import Path

_V2_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = _V2_DIR / "mining_bot_v2.log.txt"

_log_file = None


def init_log() -> Path:
  global _log_file
  if _log_file is None:
    _log_file = LOG_PATH.open("a", encoding="utf-8")
    _log_file.write(f"\n=== sessao {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    _log_file.flush()
  return LOG_PATH


def mlog(message: str) -> None:
  line = message.rstrip("\n")
  print(line, flush=True)
  if _log_file is None:
    init_log()
  assert _log_file is not None
  _log_file.write(line + "\n")
  _log_file.flush()


def close_log() -> None:
  global _log_file
  if _log_file is not None:
    _log_file.flush()
    _log_file.close()
    _log_file = None
