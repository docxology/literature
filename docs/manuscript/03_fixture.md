# Synthetic metadata replay

The checked-in fixture contains two distinct invented titles. One appears
twice with the same placeholder DOI and different source labels. The replay
submits all three records through the reference manager into a fresh
temporary library. The duplicate returns the first record's key, and the
other title receives a different key.

A new library object opens the stored index and exports it as JSON. There
are two entries after reload, and the BibTeX keys match the index keys.
Figure @fig:metadata-round-trip presents counts from the staged fixture
report. No PDFs were downloaded. The source labels describe the fixture and
do not identify provider observations.

![Three fixture submissions, two index records after reload, and two BibTeX entries.](../../output/figures/metadata-round-trip.png){#fig:metadata-round-trip width=85%}
