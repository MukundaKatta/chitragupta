# Architecture

## Overview

Chitragupta implements a modular RAG (Retrieval-Augmented Generation) pipeline with the following stages:

```
Documents -> Ingest -> Chunk -> Embed -> Index -> Search
```

## Modules

### core.py
- `Document` and `Chunk` dataclasses
- `RAGPipeline` orchestrator that ties all stages together

### chunker.py
- `FixedSizeChunker` — character-window splitting with overlap
- `SentenceChunker` — sentence-boundary splitting
- `ParagraphChunker` — paragraph-boundary splitting with short-paragraph merging

### embedder.py
- `EmbeddingSimulator` — deterministic hash-based pseudo-embeddings (no ML deps)
- `cosine_similarity` — vector similarity computation
- `EmbeddingCache` — text-keyed deduplication cache

### search.py
- `SemanticSearch` — brute-force cosine similarity search
- `SearchResult` — ranked result with score and metadata

### config.py
- `PipelineConfig` — environment-variable-driven configuration

### cli.py
- `ingest`, `search`, `stats` sub-commands
