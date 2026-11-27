# Search and local metadata

The repository provides a standalone workflow for literature discovery,
reference tracking, PDF handling, and optional analysis [@literatureSoftware].
Search adapters return a shared record type. Its metadata fields include a
title, author list, source, and optional year, DOI, and PDF location. Provider
ranking and hit availability depend on the query and service. Saving metadata
is a separate operation that can be exercised with local records.

A DOI can identify a digital object [@doiHandbook]. The fixture uses a
placeholder identifier string to test equality and never resolves it. A
bibliographic citation key is a local handle used in the JSON index and
BibTeX, a format used by the BibTeX tool [@bibtex]. The key can identify an
entry even when DOI metadata is absent.
