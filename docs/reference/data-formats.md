# Data Formats Reference

Complete documentation of data structures and file formats.

## Library Index (`library.json`)

JSON-based index of all papers in the library.

### Structure

```json
{
  "version": "1.0",
  "updated": "2025-12-02T04:42:16.615302",
  "count": 499,
  "entries": {
    "citation_key": {
      "citation_key": "smith2024machine",
      "title": "Paper Title",
      "authors": ["Author 1", "Author 2"],
      "year": 2024,
      "doi": "10.1234/example.doi",
      "source": "arxiv",
      "url": "http://arxiv.org/abs/2401.00001",
      "pdf_path": "data/pdfs/smith2024machine.pdf",
      "added_date": "2025-12-01T10:00:00.000000",
      "abstract": "Abstract text...",
      "venue": "arXiv preprint",
      "citation_count": 42
    }
  }
}
```

### Fields

- `citation_key` (str): Unique identifier
- `title` (str): Paper title
- `authors` (List[str]): Author list
- `year` (int): Publication year
- `doi` (str, optional): Digital Object Identifier
- `source` (str): Source database
- `url` (str): Paper URL
- `pdf_path` (str, optional): Path to PDF file
- `added_date` (str): ISO format timestamp
- `abstract` (str): Paper abstract
- `venue` (str, optional): Publication venue
- `citation_count` (int, optional): Citation count

## Bibliography (`references.bib`)

Standard BibTeX format bibliography.

### Format

```bibtex
@article{citation_key,
  title={Paper Title},
  author={Author1, First and Author2, Second},
  journal={Journal Name},
  year={2024},
  doi={10.1234/example.doi},
  url={http://arxiv.org/abs/2401.00001}
}
```

## Summary Files (`summaries/{citation_key}_summary.md`)

Markdown format summaries with metadata.

### Structure

```markdown
# Paper Summary: Paper Title

## Generated Summary

[Summary text here...]

## Metadata

- **Citation Key**: citation_key
- **Generated**: 2025-12-02T10:00:00
- **Model**: llama3.2:3b
- **Quality Score**: 0.85
- **Validation Status**: Accepted
- **Input Words**: 15,234
- **Output Words**: 1,247
- **Generation Time**: 45.2s
```

## Progress File (`summarization_progress.json`)

Progress tracking for summarization operations.

### Structure

```json
{
  "version": "1.0",
  "updated": "2025-12-02T10:00:00",
  "total": 10,
  "completed": 7,
  "failed": 1,
  "skipped": 2,
  "entries": {
    "citation_key": {
      "status": "completed",
      "progress": 1.0,
      "summary_path": "data/summaries/citation_key_summary.md"
    }
  }
}
```

## Meta-Analysis Outputs

### Summary JSON (`meta_analysis_summary.json`)

```json
{
  "total_papers": 499,
  "year_range": {"min": 2010, "max": 2024},
  "sources": {"arxiv": 185, "semanticscholar": 142},
  "keywords": {"machine learning": 45, "deep learning": 32}
}
```

### PCA Loadings (`pca_loadings.json`)

```json
{
  "components": 3,
  "explained_variance": [0.45, 0.23, 0.12],
  "loadings": {
    "component_0": {"word1": 0.85, "word2": 0.72},
    "component_1": {"word3": 0.91, "word4": 0.68}
  }
}
```

## See Also

- **[Data Directory Documentation](../data/AGENTS.md)** - Complete data documentation
- **[API Reference](api-reference.md)** - API documentation




