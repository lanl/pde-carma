"""Allow ``python -m pde_carma`` to use the unified launcher."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())

