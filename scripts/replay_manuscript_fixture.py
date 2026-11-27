"""Run the manuscript's offline bibliographic metadata fixture."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infrastructure.literature.library.replay_fixture import replay_fixture


if __name__ == "__main__":
    replay_fixture(ROOT / "tests" / "fixtures" / "bibliographic_metadata.json", ROOT / "output")
