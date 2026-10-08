import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vraven.figures.render import main

if __name__ == "__main__":
    raise SystemExit(main(["--figure", "all", "--mode", "all", "--formats", "svg,png,pdf", "--output", str(ROOT / "exports"), "--png-scale", "2.0"]))
