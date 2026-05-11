"""Entry point so users can run the SR CLI as ``python -m src.sr``.

Why this file exists:
    Python looks for ``<package>/__main__.py`` when invoked with ``python -m``.
    Keeping the actual argparse logic in ``cli.py`` lets us unit-test it
    without going through the module-execution path.
"""

from src.sr.cli import main

if __name__ == "__main__":
    main()
