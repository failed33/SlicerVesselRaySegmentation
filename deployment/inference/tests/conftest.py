"""Source-checkout tests; wheel acceptance tests use an installed package."""

import os
import sys
from pathlib import Path

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
