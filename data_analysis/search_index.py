"""
Search index for parquet files using Xapian.
This module provides utilities for:
- Building a full-text search index from parquet files
- Performing fuzzy searches across the indexed dataset
- Managing the index in the nanochat base directory

The index is stored in ~/.cache/nanochat/search_index/ by default.
"""

import os
import argparse
import logging
import time
import xapian
import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files, DATA_DIR

# Setup logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Index location configuration

def get_index_dir():
    """Returns the directory where the search index is stored."""
    base_dir = get_base_dir()
    index_dir = os.path.join(base_dir, "search_index")
    os.makedirs(index_dir, exist_ok=True)
    return index_dir

# -----------------------------------------------------------------------------
# Index building functions

def build_index(data_dir=None, index_dir=None):
    """
    Build a Xapian search index from all parquet files.
    
    Args:
        data_dir: Directory containing parquet files. If None, uses DATA_DIR from dataset.py
        index_dir: Directory to store the index. If None, uses get_index_dir()
    
    Returns:
        Tuple of (index_dir, total_documents_indexed)
    """
    if index_dir is None:
        index_dir = get_index_dir()
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    logger.info(f"Building search index from data in: {data_dir}")
    logger.info(f"Index will be stored in: {index_dir}")
    
    # Create or open the database
    logger.info("Creating Xapian database...")
    database = xapian.WritableDatabase(index_dir, xapian.DB_CREATE_OR_OVERWRITE)
    
    # Create term generator for indexing
    termgenerator = xapian.TermGenerator()
    termgenerator.set_stemmer(xapian.Stem("en"))
    
    # Get all parquet files
    parquet_paths = list_parquet_files(data_dir)
    logger.info(f"Found {len(parquet_paths)} parquet files to index")
    
    if len(parquet_paths) == 0:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    total_docs = 0
    start_time = time.time()
    
    # Index each parquet file
    for file_idx, filepath in enumerate(parquet_paths):
        file_start_time = time.time()
        logger.info(f"Processing file {file_idx + 1}/{len(parquet_paths)}: {os.path.basename(filepath)}")
        
        try:
            pf = pq.ParquetFile(filepath)
            num_row_groups = pf.num_row_groups
            logger.info(f"  File has {num_row_groups} row groups")
            
            file_doc_count = 0
            for rg_idx in range(num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                
                for doc_idx, text in enumerate(texts):
                    # Create a Xapian document
                    doc = xapian.Document()
                    
                    # Store metadata about document location
                    # Value slots: 0=file_idx, 1=rg_idx, 2=doc_idx
                    doc.add_value(0, str(file_idx))
                    doc.add_value(1, str(rg_idx))
                    doc.add_value(2, str(doc_idx))
                    
                    # Index the text content
                    termgenerator.set_document(doc)
                    termgenerator.index_text(text)
                    
                    # Add to database
                    database.add_document(doc)
                    total_docs += 1
                    file_doc_count += 1
                    
                    # Log progress every 10000 documents
                    if total_docs % 10000 == 0:
                        elapsed = time.time() - start_time
                        docs_per_sec = total_docs / elapsed if elapsed > 0 else 0
                        logger.info(f"  Indexed {total_docs} documents ({docs_per_sec:.1f} docs/sec)")
            
            file_elapsed = time.time() - file_start_time
            logger.info(f"  Completed file {file_idx + 1}: {file_doc_count} documents in {file_elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
            continue
    
    # Commit changes
    logger.info("Committing index to disk...")
    database.commit()
    database.close()
    
    total_elapsed = time.time() - start_time
    logger.info(f"Index building complete!")
    logger.info(f"Total documents indexed: {total_docs}")
    logger.info(f"Total time: {total_elapsed:.1f}s ({total_docs / total_elapsed:.1f} docs/sec)")
    logger.info(f"Index location: {index_dir}")
    
    return index_dir, total_docs


def search(query_text, index_dir=None, max_results=10, fuzzy=True, data_dir=None):
    """
    Search the index for documents matching the query.
    
    Args:
        query_text: The text to search for
        index_dir: Directory containing the index. If None, uses get_index_dir()
        max_results: Maximum number of results to return
        fuzzy: If True, enables fuzzy matching (typo tolerance)
        data_dir: Directory containing parquet files (needed if text not stored in index)
    
    Returns:
        List of dictionaries with keys:
            - file_idx: Index of the parquet file
            - rg_idx: Row group index within the file
            - doc_idx: Document index within the row group
            - text: The document text
            - score: Relevance score (0-1)
            - rank: Result rank (1-based)
    """
    if index_dir is None:
        index_dir = get_index_dir()
    
    if not os.path.exists(index_dir):
        raise FileNotFoundError(
            f"Index directory does not exist: {index_dir}. "
            f"Please build the index first using build_index()"
        )
    
    logger.info(f"Searching index at: {index_dir}")
    logger.info(f"Query: '{query_text}'")
    logger.info(f"Fuzzy matching: {fuzzy}, Max results: {max_results}")
    
    try:
        # Open the database for reading
        database = xapian.Database(index_dir)
        logger.info(f"Index contains {database.get_doccount()} documents")
        
        # Create query parser
        queryparser = xapian.QueryParser()
        queryparser.set_stemmer(xapian.Stem("en"))
        queryparser.set_database(database)
        
        # Enable fuzzy matching flags
        flags = xapian.QueryParser.FLAG_DEFAULT
        if fuzzy:
            # FLAG_SPELLING_CORRECTION allows fuzzy matching
            flags |= xapian.QueryParser.FLAG_SPELLING_CORRECTION
            # FLAG_PARTIAL allows partial word matching
            flags |= xapian.QueryParser.FLAG_PARTIAL
        
        # Parse the query
        query = queryparser.parse_query(query_text, flags)
        logger.debug(f"Parsed query: {query}")
        
        # Perform the search
        enquire = xapian.Enquire(database)
        enquire.set_query(query)
        
        # Get results
        search_start = time.time()
        matches = enquire.get_mset(0, max_results)
        search_time = time.time() - search_start
        
        logger.info(f"Found {matches.get_matches_estimated()} matching documents")
        logger.info(f"Returning top {len(matches)} results (search took {search_time:.3f}s)")
        
        # Extract results
        results = []
        for match in matches:
            doc = match.document
            
            # Extract metadata
            file_idx = int(doc.get_value(0))
            rg_idx = int(doc.get_value(1))
            doc_idx = int(doc.get_value(2))
            
            # Retrieve text from parquet file
            if data_dir is None:
                data_dir = DATA_DIR
            text = retrieve_text_from_parquet(file_idx, rg_idx, doc_idx, data_dir)
            
            # Get relevance score (as percentage, convert to 0-1)
            score = match.percent / 100.0
            
            result = {
                'file_idx': file_idx,
                'rg_idx': rg_idx,
                'doc_idx': doc_idx,
                'text': text,
                'score': score,
                'rank': match.rank + 1
            }
            results.append(result)
            
            logger.debug(f"  Result {match.rank + 1}: score={score:.2%}, location=({file_idx},{rg_idx},{doc_idx})")
        
        database.close()
        return results
        
    except xapian.DatabaseOpeningError as e:
        raise FileNotFoundError(
            f"Failed to open index: {e}. "
            f"Please build the index first using build_index()"
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise


def retrieve_text_from_parquet(file_idx, rg_idx, doc_idx, data_dir=None):
    """
    Retrieve the original text from a parquet file given its location.
    
    Args:
        file_idx: Index of the parquet file
        rg_idx: Row group index
        doc_idx: Document index within the row group
        data_dir: Directory containing parquet files
    
    Returns:
        The text content as a string, or None if not found
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    try:
        parquet_paths = list_parquet_files(data_dir)
        if file_idx >= len(parquet_paths):
            raise IndexError(f"Invalid file_idx: {file_idx} (only {len(parquet_paths)} files)")
        
        filepath = parquet_paths[file_idx]
        pf = pq.ParquetFile(filepath)
        
        if rg_idx >= pf.num_row_groups:
            raise IndexError(f"Invalid rg_idx: {rg_idx} (only {pf.num_row_groups} row groups)")
        
        rg = pf.read_row_group(rg_idx)
        texts = rg.column('text').to_pylist()
        
        if doc_idx >= len(texts):
            raise IndexError(f"Invalid doc_idx: {doc_idx} (only {len(texts)} documents)")
        
        return texts[doc_idx]
        
    except Exception as e:
        logger.error(f"Error retrieving text from parquet: {e}")
        raise


def get_index_stats(index_dir=None):
    """
    Get statistics about the search index.
    
    Args:
        index_dir: Directory containing the index. If None, uses get_index_dir()
    
    Returns:
        Dictionary with index statistics, or None if index doesn't exist
    """
    if index_dir is None:
        index_dir = get_index_dir()
    
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Index directory does not exist: {index_dir}")
    
    try:
        database = xapian.Database(index_dir)
        
        stats = {
            'index_dir': index_dir,
            'num_documents': database.get_doccount(),
            'last_docid': database.get_lastdocid(),
            'num_terms': database.get_total_term_freq(),
            'avg_doc_length': database.get_avlength(),
        }
        
        database.close()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        raise


# -----------------------------------------------------------------------------
# Command-line interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and search a full-text index of parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build the index
  python search_index.py --build
  
  # Search the index
  python search_index.py --search "Tarun loves banana"
  
  # Get index statistics
  python search_index.py --stats
        """
    )
    
    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--build", action="store_true", help="Build the search index")
    action_group.add_argument("--search", type=str, metavar="QUERY", help="Search the index")
    action_group.add_argument("--stats", action="store_true", help="Show index statistics")
    
    # Common arguments
    parser.add_argument("--index-dir", type=str, help="Custom index directory (default: uses get_index_dir())")
    parser.add_argument("--data-dir", type=str, help="Custom data directory (default: uses DATA_DIR)")
    
    # Search arguments
    parser.add_argument("-n", "--max-results", type=int, default=10, 
                        help="Maximum number of search results (default: 10)")
    parser.add_argument("--no-fuzzy", action="store_true", 
                        help="Disable fuzzy matching (exact matches only)")
    parser.add_argument("--preview-length", type=int, default=200,
                        help="Length of text preview in results (default: 200)")
    
    args = parser.parse_args()
    
    # Set up logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Execute the requested action
    if args.build:
        logger.info("="*80)
        logger.info("Building search index")
        logger.info("="*80)
        index_dir, total_docs = build_index(
            data_dir=args.data_dir,
            index_dir=args.index_dir
        )
        logger.info("="*80)
        logger.info(f"Index built successfully with {total_docs} documents")
        logger.info(f"Index location: {index_dir}")
        
    elif args.search:
        logger.info("="*80)
        logger.info("Searching index")
        logger.info("="*80)
        results = search(
            query_text=args.search,
            index_dir=args.index_dir,
            max_results=args.max_results,
            fuzzy=not args.no_fuzzy,
            data_dir=args.data_dir
        )
        
        logger.info("="*80)
        logger.info(f"Search Results ({len(results)} matches)")
        logger.info("="*80)
        
        for result in results:
            print(f"\nRank #{result['rank']} - Score: {result['score']:.1%}")
            print(f"Location: file {result['file_idx']}, row_group {result['rg_idx']}, doc {result['doc_idx']}")
            
            # Show text preview
            text = result['text']
            if len(text) > args.preview_length:
                preview = text[:args.preview_length] + "..."
            else:
                preview = text
            print(f"Text preview:\n{preview}")
            print("-" * 80)
        
        if len(results) == 0:
            logger.warning("No results found. Try different search terms or enable fuzzy matching.")
    
    elif args.stats:
        logger.info("="*80)
        logger.info("Index Statistics")
        logger.info("="*80)
        stats = get_index_stats(index_dir=args.index_dir)
        
        print(f"\nIndex Directory: {stats['index_dir']}")
        print(f"Total Documents: {stats['num_documents']:,}")
        print(f"Last Document ID: {stats['last_docid']:,}")
        print(f"Total Terms: {stats['num_terms']:,}")
        print(f"Average Document Length: {stats['avg_doc_length']:.1f} terms")

