from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pde_carma.launcher import wrapper_main

if __name__ == "__main__":
    raise SystemExit(wrapper_main("flowformer-unidirectional", "infer"))

