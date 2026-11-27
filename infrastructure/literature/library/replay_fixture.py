"""Exercise the local reference and index APIs on authored synthetic metadata."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from infrastructure.literature.core.config import LiteratureConfig
from infrastructure.literature.library.index import LibraryIndex
from infrastructure.literature.library.references import ReferenceManager
from infrastructure.literature.sources.base import SearchResult

FIGURE = "metadata-round-trip.png"
ALT = "Three submissions, two records after the restart, and two BibTeX entries."


def replay_fixture(fixture: Path, output: Path) -> None:
    """Build a traceable count report and plot without contacting providers."""
    fixture_bytes = fixture.read_bytes()
    records = json.loads(fixture_bytes)
    fixture_output = output / "fixture"
    figures = output / "figures"
    reports = output / "reports"
    for path in (fixture_output, figures, reports):
        path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        config = LiteratureConfig(
            download_dir=str(scratch / "pdfs"),
            library_index_file=str(scratch / "library.json"),
            bibtex_file=str(scratch / "references.bib"),
            use_unpaywall=False,
        )
        index = LibraryIndex(config)
        references = ReferenceManager(config, index)
        keys = [references.add_reference(SearchResult(**record)) for record in records]
        reloaded = LibraryIndex(config)
        export = reloaded.export_json(scratch / "export.json")
        bibtex = Path(config.bibtex_file).read_text(encoding="utf-8")
        bibtex_keys = re.findall(r"^@\w+\{([^,]+),", bibtex, re.MULTILINE)
        exported_entries = json.loads(export.read_text(encoding="utf-8"))["entries"]
        assert keys[0] == keys[1] and keys[1] != keys[2]
        assert {record.citation_key for record in reloaded.list_entries()} == set(keys)
        assert len(bibtex_keys) == len(exported_entries) == len(set(keys))
        assert set(bibtex_keys) == {entry["citation_key"] for entry in exported_entries} == set(keys)
        assert all(record.pdf_path is None for record in reloaded.list_entries())
        for filename in ("library.json", "references.bib", export.name):
            shutil.copyfile(scratch / filename, fixture_output / filename)

    counts = {
        "fixture_source": fixture.name,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "submitted_records": len(records),
        "reloaded_library_entries": len(reloaded.list_entries()),
        "bibtex_entries": len(bibtex_keys),
        "duplicate_pairs_same_key": int(keys[0] == keys[1]),
        "downloaded_pdfs": sum(bool(entry.pdf_path) for entry in reloaded.list_entries()),
    }
    (reports / "metadata-fixture.json").write_text(
        json.dumps(counts, indent=2) + "\n", encoding="utf-8"
    )

    fig = Figure(figsize=(7, 3.3), layout="constrained")
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    axis.barh(
        ["Submitted metadata", "Index after reload", "BibTeX export"],
        [counts["submitted_records"], counts["reloaded_library_entries"], counts["bibtex_entries"]],
        color=["#3c6997", "#29786d", "#29786d"],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Records")
    axis.set_xticks(range(len(records) + 1))
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / FIGURE, dpi=160)

    registry = {
        "schema_version": 1,
        "figures": [{
            "label": "fig:metadata-round-trip",
            "filename": FIGURE,
            "caption": "Counts from the synthetic library replay.",
            "alt_text": ALT,
            "generated_by": "scripts/replay_manuscript_fixture.py",
        }],
    }
    (figures / "figure_registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
