# tests/conftest.py
#
# Pytest configuration: adds src/ to sys.path so `import tilegen` works
# when running the suite directly from the repo

import sys
from pathlib import Path

# Make `import tilegen` work when running tests directly from the repo without
# an installed (editable) package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
