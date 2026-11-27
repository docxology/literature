# Index, bibliography, and export

With a library index attached, the reference manager first attempts to store
an index entry and obtains its citation key. The index checks supplied DOI
equality, then exact title equality after removing surrounding whitespace and
folding case. It reuses an existing key on a match. Otherwise, author, year, and
title information help generate a new key. This is a local, heuristic identity
rule. It depends on the supplied metadata and can miss variants or conflate
distinct works with the same title.

The persisted index stores entries keyed by their citation keys. The JSON
export presents them as a list. Both can include times assigned by the
writer. Re-exporting the same entries does not imply byte-identical JSON.
The reference manager writes a separate BibTeX file; index and bibliography
writes are separate operations. Compare keys and contents when checking a
replay. A PDF path, when present, records an association with the library
entry.
