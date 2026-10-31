"""Run the unified pde-carma launcher from a clean checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pde_carma.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

