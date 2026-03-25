# Contributing to Chitragupta

## Getting Started

```bash
git clone https://github.com/MukundaKatta/chitragupta.git
cd chitragupta
```

## Running Tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

Or use Make:

```bash
make test
```

## Code Style

- Python 3.9+ compatible (use `from __future__ import annotations`)
- No external dependencies for core functionality
- Type hints on all public functions
- Docstrings on all public classes and methods

## Adding a New Chunking Strategy

1. Create a class in `src/chitragupta/chunker.py` extending `ChunkerBase`
2. Implement the `chunk(text) -> List[str]` method
3. Register the strategy name in `RAGPipeline.VALID_STRATEGIES`
4. Add tests in `tests/test_chunker.py`
