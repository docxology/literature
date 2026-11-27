"""Check the authored fixture against the exported index, BibTeX, and figure."""

import hashlib
import json
from pathlib import Path

from infrastructure.literature.library.replay_fixture import replay_fixture


def test_replay_metadata_fixture(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "bibliographic_metadata.json"
    replay_fixture(fixture, tmp_path)
    report = json.loads((tmp_path / "reports" / "metadata-fixture.json").read_text())
    assert report == {
        "fixture_source": fixture.name,
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "submitted_records": 3,
        "reloaded_library_entries": 2,
        "bibtex_entries": 2,
        "duplicate_pairs_same_key": 1,
        "downloaded_pdfs": 0,
    }
    export = json.loads((tmp_path / "fixture" / "export.json").read_text())
    assert len(export["entries"]) == 2
    assert (tmp_path / "figures" / "metadata-round-trip.png").stat().st_size > 0
