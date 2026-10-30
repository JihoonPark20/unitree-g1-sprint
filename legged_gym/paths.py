from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
DEFAULT_PRIOR = ROOT_DIR / "outputs" / "prior.pt"
