"""
Data analysis utilities for nanochat.

This package provides tools for analyzing and searching the training data:
- search_index: Full-text search index using Xapian for fuzzy searching across parquet files
- check_contamination: Dataset contamination detection for GSM8K and MATH datasets
"""

from .search_index import build_index, search, get_index_stats, get_index_dir, SearchContext

__all__ = ['build_index', 'search', 'get_index_stats', 'get_index_dir', 'SearchContext']

