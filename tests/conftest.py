"""Makes the modules in src/ importable in the tests, as when running python src/<file>.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
