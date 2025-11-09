"""
Data analysis utilities for nanochat.

This package provides tools for analyzing and searching the training data:
- search_index: Full-text search index using Xapian for fuzzy searching across parquet files
"""

from .search_index import build_index, search, get_index_stats, get_index_dir

__all__ = ['build_index', 'search', 'get_index_stats', 'get_index_dir']

