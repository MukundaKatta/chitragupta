"""
Command-line interface for the Chitragupta RAG pipeline.

Provides ``ingest``, ``search``, and ``stats`` sub-commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from chitragupta.config import PipelineConfig
from chitragupta.core import Document, RAGPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chitragupta",
        description="Chitragupta - Document Indexing and RAG Pipeline",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingest documents from text files")
    ingest_p.add_argument(
        "files", nargs="+", help="Paths to text files to ingest"
    )
    ingest_p.add_argument(
        "--strategy",
        choices=("fixed", "sentence", "paragraph"),
        default=None,
        help="Chunking strategy (overrides config)",
    )

    # search
    search_p = sub.add_parser("search", help="Search ingested documents")
    search_p.add_argument("query", help="Search query string")
    search_p.add_argument(
        "files", nargs="+", help="Paths to text files to search over"
    )
    search_p.add_argument(
        "-k", "--top-k", type=int, default=None, help="Number of results"
    )
    search_p.add_argument(
        "--strategy",
        choices=("fixed", "sentence", "paragraph"),
        default=None,
        help="Chunking strategy",
    )

    # stats
    stats_p = sub.add_parser("stats", help="Show pipeline statistics")
    stats_p.add_argument(
        "files", nargs="*", help="Paths to text files (optional)"
    )

    return parser


def _load_documents(paths: List[str]) -> List[Document]:
    """Read text files and return Document objects."""
    docs = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            if content.strip():
                docs.append(Document(content=content, source=path))
        except OSError as exc:
            print("Warning: could not read {}: {}".format(path, exc), file=sys.stderr)
    return docs


def _make_pipeline(cfg: PipelineConfig, strategy: Optional[str] = None) -> RAGPipeline:
    return RAGPipeline(
        chunk_strategy=strategy or cfg.chunk_strategy,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        embedding_dim=cfg.embedding_dim,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = PipelineConfig.from_env()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "ingest":
        docs = _load_documents(args.files)
        if not docs:
            print("No valid documents found.", file=sys.stderr)
            return 1
        pipeline = _make_pipeline(cfg, args.strategy)
        added = pipeline.ingest(docs)
        chunks = pipeline.chunk()
        embedded = pipeline.embed()
        print("Ingested {} document(s), created {} chunk(s), embedded {}.".format(
            added, len(chunks), embedded
        ))
        return 0

    if args.command == "search":
        docs = _load_documents(args.files)
        if not docs:
            print("No valid documents found.", file=sys.stderr)
            return 1
        pipeline = _make_pipeline(cfg, args.strategy)
        top_k = args.top_k or cfg.search_top_k
        results = pipeline.run(docs, args.query, k=top_k)
        for i, r in enumerate(results, 1):
            print("{}. [score={:.4f}] {}".format(i, r.score, r.text[:120]))
        if not results:
            print("No results found.")
        return 0

    if args.command == "stats":
        pipeline = _make_pipeline(cfg)
        if args.files:
            docs = _load_documents(args.files)
            if docs:
                pipeline.run(docs, "dummy query")
        print(json.dumps(pipeline.stats(), indent=2))
        return 0

    parser.print_help()
    return 0
