# Chitragupta -- RAG Pipeline

> Hindu Mythology: The Divine Record Keeper | Document indexing and semantic search for RAG applications

[![CI](https://github.com/MukundaKatta/chitragupta/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/chitragupta/actions)
[![GitHub Pages](https://img.shields.io/badge/Live_Demo-Visit_Site-blue?style=for-the-badge)](https://MukundaKatta.github.io/chitragupta/)
[![GitHub](https://img.shields.io/github/license/MukundaKatta/chitragupta?style=flat-square)](LICENSE)

Chitragupta is a pure-Python RAG (Retrieval-Augmented Generation) pipeline that handles document ingestion, chunking, embedding simulation, and semantic search -- with zero external dependencies.

## Features

- **Document ingestion** with automatic deduplication
- **Three chunking strategies**: fixed-size windows, sentence-boundary, paragraph-boundary
- **Hash-based pseudo-embeddings** (deterministic, no ML libraries needed)
- **Semantic search** with cosine similarity and metadata filtering
- **CLI** with `ingest`, `search`, and `stats` commands
- **Python 3.9+** compatible

## Quick Start

```python
from chitragupta import RAGPipeline, Document

pipeline = RAGPipeline(chunk_strategy="sentence", embedding_dim=64)

docs = [
    Document(content="The Ganges is a sacred river in India.", metadata={"topic": "geography"}),
    Document(content="Python is widely used for data science.", metadata={"topic": "tech"}),
]

results = pipeline.run(docs, query="rivers of India", k=3)
for r in results:
    print(f"[{r.score:.4f}] {r.text}")
```

## CLI Usage

```bash
# Ingest documents
PYTHONPATH=src python -m chitragupta ingest doc1.txt doc2.txt

# Search
PYTHONPATH=src python -m chitragupta search "sacred river" doc1.txt doc2.txt

# Pipeline stats
PYTHONPATH=src python -m chitragupta stats doc1.txt
```

## Project Structure

```
chitragupta/
├── src/chitragupta/
│   ├── __init__.py
│   ├── core.py          # RAGPipeline, Document, Chunk
│   ├── chunker.py       # FixedSize, Sentence, Paragraph chunkers
│   ├── embedder.py      # Hash-based pseudo-embeddings
│   ├── search.py        # Semantic search engine
│   ├── config.py        # Environment-based configuration
│   ├── cli.py           # Command-line interface
│   └── __main__.py
├── tests/
│   ├── test_core.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_search.py
├── docs/ARCHITECTURE.md
├── pyproject.toml
└── Makefile
```

## Running Tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Live Demo

Visit the landing page: **https://MukundaKatta.github.io/chitragupta/**

## License

MIT License -- Part of the [Mythological Portfolio](https://github.com/MukundaKatta) by Officethree Technologies.
