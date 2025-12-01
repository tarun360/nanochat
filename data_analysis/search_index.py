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
import json
import tempfile
import shutil
import xapian
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files, DATA_DIR

# Setup logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Index location configuration

def get_index_dir():
    """Returns the directory where the search index is stored."""
    base_dir = get_base_dir()
    index_dir = os.path.join(base_dir, "base_data_search_index")
    os.makedirs(index_dir, exist_ok=True)
    return index_dir

# -----------------------------------------------------------------------------
# Search context class for efficient batch searching

class SearchContext:
    """
    Context manager for efficient batch searching.
    Opens all shards once and keeps them open for multiple queries.
    
    Usage:
        with SearchContext() as ctx:
            for query in queries:
                results = ctx.search_phrase(query, max_results=10)
    """
    
    def __init__(self, index_dir=None, data_dir=None):
        self.index_dir = index_dir if index_dir else get_index_dir()
        self.data_dir = data_dir
        self.database = None
        self.num_docs = 0
        
    def __enter__(self):
        """Open all shards and prepare for searching."""
        shards_dir = os.path.join(self.index_dir, "shards")
        
        if not os.path.exists(shards_dir):
            raise FileNotFoundError(f"Shards directory does not exist: {shards_dir}")
        
        # Create combined database
        self.database = xapian.Database()
        
        # Find and add all shards
        shard_names = sorted([d for d in os.listdir(shards_dir) if d.startswith("shard_")])
        
        if len(shard_names) == 0:
            raise FileNotFoundError(f"No shards found in {shards_dir}")
        
        for shard_name in shard_names:
            shard_path = os.path.join(shards_dir, shard_name)
            self.database.add_database(xapian.Database(shard_path))
        
        # Count total documents
        for shard_name in shard_names:
            shard_path = os.path.join(shards_dir, shard_name)
            shard_db = xapian.Database(shard_path)
            self.num_docs += shard_db.get_doccount()
            shard_db.close()
        
        logger.debug(f"SearchContext opened: {self.num_docs:,} documents across {len(shard_names)} shards")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the database."""
        if self.database:
            self.database.close()
        return False
    
    def search_phrase(self, phrase_text, max_results=10, slop=20):
        """
        Search for a phrase using OP_PHRASE.
        
        This is the preferred search method because it robustly handles special characters,
        punctuation, and mathematical notation (e.g., "$(8,-8)$", "3^x") that would break
        QueryParser-based search. It uses TermGenerator to process the query text the same
        way it was indexed, ensuring consistent tokenization, stemming, and term extraction.
        
        Args:
            phrase_text: The phrase to search for (can contain special characters)
            max_results: Maximum number of results to return
            slop: Extra distance allowed between terms (default: 20)
                  Window size will be len(terms) + slop
        
        Returns:
            List of result dictionaries with keys:
                - file_idx: Index of the parquet file
                - rg_idx: Row group index within the file
                - doc_idx: Document index within the row group
                - text: The document text
                - score: Relevance score (0-1)
                - rank: Result rank (1-based)
        """
        if not self.database:
            raise RuntimeError("SearchContext not initialized. Use 'with SearchContext() as ctx:'")
        
        # Use TermGenerator to process query text the same way as indexing
        tg = xapian.TermGenerator()
        tg.set_stemmer(xapian.Stem("en"))
        
        # Create temporary document and index the query text
        tmp = xapian.Document()
        tg.set_document(tmp)
        tg.index_text(phrase_text)
        
        # Extract terms in position order
        pos_to_terms = {}
        for titem in tmp.termlist():
            termname = titem.term
            pos_iter = titem.positer
            for pos in pos_iter:
                pos_to_terms.setdefault(pos, []).append(termname)
        
        if not pos_to_terms:
            logger.warning(f"No valid terms from phrase: {phrase_text[:100]}")
            return []
        
        # Build term sequence by position order
        sequence = []
        for pos in sorted(pos_to_terms.keys()):
            sequence.extend(pos_to_terms[pos])
        
        # Calculate window: base phrase length + allowed slop
        window = len(sequence) + slop
        
        # Construct OP_PHRASE query from the term sequence
        if len(sequence) == 1:
            query = xapian.Query(sequence[0])
        else:
            query = xapian.Query(xapian.Query.OP_PHRASE, sequence, window)
        
        logger.debug(f"Phrase query: {len(sequence)} terms, slop={slop}, window={window}")
        
        # Perform search
        enquire = xapian.Enquire(self.database)
        enquire.set_query(query)
        matches = enquire.get_mset(0, max_results)
        
        # Extract results
        results = []
        for match in matches:
            doc = match.document
            file_idx = int(doc.get_value(0))
            rg_idx = int(doc.get_value(1))
            doc_idx = int(doc.get_value(2))
            text = retrieve_text_from_parquet(file_idx, rg_idx, doc_idx, self.data_dir)
            score = match.percent / 100.0
            
            results.append({
                'file_idx': file_idx,
                'rg_idx': rg_idx,
                'doc_idx': doc_idx,
                'text': text,
                'score': score,
                'rank': match.rank + 1
            })
        
        return results
    
    def search_phrase_ngrams(self, phrase_text, ngram_size=10, num_ngrams=3, max_results=10, slop=5):
        """
        Search for a phrase using n-gram matching.
        
        Extracts multiple n-grams (e.g., start, middle, end) from the phrase after
        TermGenerator processing, then searches for each. Returns the maximum score
        across all n-grams and details about which n-grams matched.
        
        This is useful for detecting partial contamination when the full phrase
        might not be present in the training data.
        
        Args:
            phrase_text: The phrase to search for (can contain special characters)
            ngram_size: Number of terms per n-gram (default: 10)
            num_ngrams: Number of n-grams to extract (default: 3)
            max_results: Maximum number of results to return per n-gram
            slop: Extra distance allowed between terms in phrase matching (default: 5)
        
        Returns:
            Dictionary with keys:
                - max_score: Maximum score across all n-grams (0-1)
                - best_match: Best result dictionary (same format as search_phrase)
                - ngram_details: List of n-gram match info with keys:
                    - position: 'start', 'middle', or 'end'
                    - matched: Boolean indicating if any match found
                    - score: Maximum score for this n-gram (0 if no match)
                    - match_count: Number of matches found
        """
        if not self.database:
            raise RuntimeError("SearchContext not initialized. Use 'with SearchContext() as ctx:'")
        
        # Use TermGenerator to process query text the same way as indexing
        tg = xapian.TermGenerator()
        tg.set_stemmer(xapian.Stem("en"))
        
        # Create temporary document and index the query text
        tmp = xapian.Document()
        tg.set_document(tmp)
        tg.index_text(phrase_text)
        
        # Extract terms in position order
        pos_to_terms = {}
        for titem in tmp.termlist():
            termname = titem.term
            pos_iter = titem.positer
            for pos in pos_iter:
                pos_to_terms.setdefault(pos, []).append(termname)
        
        if not pos_to_terms:
            logger.warning(f"No valid terms from phrase: {phrase_text[:100]}")
            return {
                'max_score': 0.0,
                'best_match': None,
                'ngram_details': []
            }
        
        # Build term sequence by position order
        sequence = []
        for pos in sorted(pos_to_terms.keys()):
            sequence.extend(pos_to_terms[pos])
        
        # Check if phrase is too short - fallback to full phrase search
        if len(sequence) <= ngram_size:
            logger.debug(f"Phrase too short ({len(sequence)} terms), using full phrase search")
            results = self.search_phrase(phrase_text, max_results=max_results, slop=slop)
            # For full phrase search, store the entire sequence as the matched n-gram
            matched_ngram_text = ' '.join(sequence) if results else None
            return {
                'max_score': results[0]['score'] if results else 0.0,
                'best_match': results[0] if results else None,
                'ngram_details': [{
                    'position': 'full',
                    'matched': len(results) > 0,
                    'score': results[0]['score'] if results else 0.0,
                    'match_count': len(results),
                    'ngram_terms': matched_ngram_text
                }]
            }
        
        # Extract n-grams from term sequence
        ngrams = []
        total_terms = len(sequence)
        
        # Extract n-grams based on num_ngrams parameter
        if num_ngrams >= 1:
            # Start n-gram
            ngrams.append(('start', sequence[0:ngram_size]))
        
        if num_ngrams >= 3:
            # Middle n-gram (only when extracting 3 or more n-grams)
            mid_start = total_terms // 2 - ngram_size // 2
            mid_start = max(0, min(mid_start, total_terms - ngram_size))
            ngrams.append(('middle', sequence[mid_start:mid_start + ngram_size]))
        
        if num_ngrams >= 2:
            # End n-gram (for 2 or more n-grams)
            ngrams.append(('end', sequence[-ngram_size:]))
        
        # Search each n-gram
        ngram_details = []
        best_score = 0.0
        best_match = None
        
        # Reuse Enquire object for all n-gram searches
        enquire = xapian.Enquire(self.database)
        
        for position, term_subsequence in ngrams:
            # Construct OP_PHRASE query from term subsequence
            if len(term_subsequence) == 1:
                query = xapian.Query(term_subsequence[0])
            else:
                window = len(term_subsequence) + slop
                query = xapian.Query(xapian.Query.OP_PHRASE, term_subsequence, window)
            
            # Perform search (reusing Enquire object)
            enquire.set_query(query)
            matches = enquire.get_mset(0, max_results)
            
            match_count = matches.size()
            max_ngram_score = 0.0
            
            # Get the first (best) match for this n-gram if any exist
            if match_count > 0:
                match = next(iter(matches))
                doc = match.document
                score = match.percent / 100.0
                max_ngram_score = score
                
                # Extract metadata (don't retrieve text yet - only for best match)
                file_idx = int(doc.get_value(0))
                rg_idx = int(doc.get_value(1))
                doc_idx = int(doc.get_value(2))
                
                # Store match info without text (lazy retrieval)
                ngram_match_info = {
                    'file_idx': file_idx,
                    'rg_idx': rg_idx,
                    'doc_idx': doc_idx,
                    'score': score,
                    'rank': match.rank + 1
                }
                
                # Update overall best match (only if this is better)
                if score > best_score:
                    best_score = score
                    # Only retrieve text for the best match
                    text = retrieve_text_from_parquet(file_idx, rg_idx, doc_idx, self.data_dir)
                    best_match = {
                        'file_idx': file_idx,
                        'rg_idx': rg_idx,
                        'doc_idx': doc_idx,
                        'text': text,
                        'score': score,
                        'rank': match.rank + 1
                    }
            
            # Record n-gram search result
            # Store the matched n-gram terms for inspection (terms are stemmed, so this shows what matched)
            matched_ngram_text = ' '.join(term_subsequence) if match_count > 0 else None
            ngram_details.append({
                'position': position,
                'matched': match_count > 0,
                'score': max_ngram_score,
                'match_count': match_count,
                'ngram_terms': matched_ngram_text  # The actual terms that matched (stemmed)
            })
        
        return {
            'max_score': best_score,
            'best_match': best_match,
            'ngram_details': ngram_details
        }


# -----------------------------------------------------------------------------
# Index building functions

def _index_worker(args):
    """
    Worker function for parallel indexing. Creates a shard of the index.
    
    Args:
        args: Tuple of (worker_id, file_paths, shard_dir, data_dir, no_sync, start_file_idx)
    
    Returns:
        Tuple of (worker_id, num_docs_indexed, shard_dir)
    """
    worker_id, file_paths, shard_dir, data_dir, no_sync, start_file_idx = args
    
    # Set up logging for worker
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    
    worker_logger.info(f"Worker {worker_id} starting: {len(file_paths)} files to process")
    
    # Create shard database
    db_flags = xapian.DB_CREATE_OR_OVERWRITE
    if no_sync:
        db_flags |= xapian.DB_NO_SYNC
    
    database = xapian.WritableDatabase(shard_dir, db_flags)
    
    # Create term generator
    termgenerator = xapian.TermGenerator()
    termgenerator.set_stemmer(xapian.Stem("en"))
    
    total_docs = 0
    start_time = time.time()
    
    try:
        for file_idx_local, filepath in enumerate(file_paths):
            filename = os.path.basename(filepath)
            file_start = time.time()
            
            # Calculate global file index
            global_file_idx = start_file_idx + file_idx_local
            
            try:
                pf = pq.ParquetFile(filepath)
                num_row_groups = pf.num_row_groups
                
                file_doc_count = 0
                for rg_idx in range(num_row_groups):
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column('text').to_pylist()
                    
                    for doc_idx, text in enumerate(texts):
                        doc = xapian.Document()
                        
                        # Store metadata with global file index
                        doc.add_value(0, str(global_file_idx))
                        doc.add_value(1, str(rg_idx))
                        doc.add_value(2, str(doc_idx))
                        
                        # Index text
                        termgenerator.set_document(doc)
                        termgenerator.index_text(text)
                        
                        database.add_document(doc)
                        total_docs += 1
                        file_doc_count += 1
                        
                        # Log progress every 10000 documents
                        if total_docs % 10000 == 0:
                            elapsed = time.time() - start_time
                            docs_per_sec = total_docs / elapsed if elapsed > 0 else 0
                            worker_logger.info(f"Worker {worker_id}: Indexed {total_docs} documents ({docs_per_sec:.1f} docs/sec)")
                
                file_elapsed = time.time() - file_start
                worker_logger.info(f"Worker {worker_id}: Completed {filename} - {file_doc_count} docs in {file_elapsed:.1f}s")
                
                # Commit after each file
                database.commit()
                
            except Exception as e:
                worker_logger.error(f"Worker {worker_id}: Error processing {filename}: {e}")
                continue
        
        # Final commit
        database.commit()
        database.close()
        
        elapsed = time.time() - start_time
        worker_logger.info(f"Worker {worker_id} finished: {total_docs} documents in {elapsed:.1f}s ({total_docs/elapsed:.1f} docs/sec)")
        
        return (worker_id, total_docs, shard_dir)
        
    except Exception as e:
        worker_logger.error(f"Worker {worker_id} failed: {e}")
        try:
            database.close()
        except:
            pass
        return (worker_id, 0, None)


def build_index(data_dir=None, index_dir=None, resume=True, no_sync=False, num_workers=8, max_files=None):
    """
    Build a Xapian search index from all parquet files using parallel workers.
    
    Args:
        data_dir: Directory containing parquet files. If None, uses DATA_DIR from dataset.py
        index_dir: Directory to store the index. If None, uses get_index_dir()
        resume: If True, resumes from previous progress. If False, starts fresh.
        no_sync: If True, disables fsync for faster indexing (riskier if system crashes)
        num_workers: Number of parallel workers to use (default: 8)
        max_files: Maximum number of parquet files to process. If None, processes all files.
    
    Returns:
        Tuple of (index_dir, total_documents_indexed)
    
    Resumability:
        - Progress is tracked at FILE level in .index_progress.json
        - On resume, already-completed files are skipped
        - Remaining files are distributed among workers
        - Progress is saved AFTER all workers finish and merge completes
        - If interrupted during worker execution:
          * Files from previous completed runs are still tracked
          * Current batch of files being processed by workers will be re-indexed
          * This is safe - just re-does some work on next run
        - Granularity: One batch of parallel processing at a time
    """
    if index_dir is None:
        index_dir = get_index_dir()
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    logger.info(f"Building search index from data in: {data_dir}")
    logger.info(f"Index will be stored in: {index_dir}")
    logger.info(f"Using {num_workers} parallel workers")
    
    # Progress tracking file
    progress_file = os.path.join(index_dir, ".index_progress.json")
    
    # Load progress if resuming
    completed_files = set()
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                completed_files = set(progress_data.get('completed_files', []))
                logger.info(f"Resuming from previous session: {len(completed_files)} files already indexed")
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}. Starting fresh.")
            completed_files = set()
    
    # Get all parquet files
    parquet_paths = list_parquet_files(data_dir)
    
    # Limit to max_files if specified
    if max_files is not None and max_files > 0:
        parquet_paths = parquet_paths[:max_files]
        logger.info(f"Limited to first {max_files} parquet files")
    
    logger.info(f"Found {len(parquet_paths)} parquet files to index")
    
    if len(parquet_paths) == 0:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    # Filter out already completed files
    remaining_files = [
        fp for fp in parquet_paths 
        if os.path.basename(fp) not in completed_files
    ]
    
    if len(remaining_files) == 0:
        logger.info("All files already indexed!")
        # Just return stats from existing index
        try:
            db = xapian.Database(index_dir)
            total_docs = db.get_doccount()
            db.close()
            return index_dir, total_docs
        except:
            return index_dir, 0
    
    logger.info(f"{len(remaining_files)} files remaining to index")
    
    # Create directory for shards within the index directory
    shards_dir = os.path.join(index_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    logger.info(f"Creating index shards in: {shards_dir}")
    
    try:
        start_time = time.time()
        
        # Divide files among workers
        files_per_worker = len(remaining_files) // num_workers
        extra_files = len(remaining_files) % num_workers
        
        worker_args = []
        file_idx = 0
        for worker_id in range(num_workers):
            # Calculate how many files this worker gets
            worker_file_count = files_per_worker + (1 if worker_id < extra_files else 0)
            worker_files = remaining_files[file_idx:file_idx + worker_file_count]
            
            if len(worker_files) == 0:
                continue
            
            shard_dir = os.path.join(shards_dir, f"shard_{worker_id}")
            os.makedirs(shard_dir, exist_ok=True)
            
            # Calculate the starting global file index for this worker
            # Need to account for all files in parquet_paths, not just remaining_files
            start_file_idx = parquet_paths.index(worker_files[0])
            
            worker_args.append((worker_id, worker_files, shard_dir, data_dir, no_sync, start_file_idx))
            file_idx += worker_file_count
            
            logger.info(f"Worker {worker_id}: {len(worker_files)} files, starting at global index {start_file_idx} ({os.path.basename(worker_files[0])} to {os.path.basename(worker_files[-1])})")
        
        # Run workers in parallel
        logger.info(f"Starting {len(worker_args)} workers...")
        with Pool(processes=len(worker_args)) as pool:
            results = pool.map(_index_worker, worker_args)
        
        # Check results
        successful_shards = [(wid, docs, shard) for wid, docs, shard in results if shard is not None]
        failed_workers = [wid for wid, docs, shard in results if shard is None]
        
        if failed_workers:
            logger.warning(f"Workers {failed_workers} failed")
        
        if len(successful_shards) == 0:
            raise RuntimeError("All workers failed!")
        
        total_docs_new = sum(docs for _, docs, _ in successful_shards)
        logger.info(f"All workers completed: {total_docs_new} new documents indexed")
        logger.info(f"Shards stored in: {shards_dir}")
        logger.info("Shards will be kept permanently for distributed searching")
        
        # Update progress file
        for fp in remaining_files:
            completed_files.add(os.path.basename(fp))
        
        try:
            with open(progress_file, 'w') as f:
                json.dump({'completed_files': list(completed_files)}, f)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
        
        total_elapsed = time.time() - start_time
        logger.info(f"Index building complete!")
        logger.info(f"Total documents indexed: {total_docs_new}")
        logger.info(f"Total files processed: {len(completed_files)}/{len(parquet_paths)}")
        logger.info(f"Total time: {total_elapsed:.1f}s ({total_docs_new / total_elapsed:.1f} docs/sec)")
        logger.info(f"Shards location: {shards_dir}")
        
        # Clean up progress file if all files are done
        if len(completed_files) == len(parquet_paths):
            logger.info("All files indexed successfully. Cleaning up progress file.")
            try:
                if os.path.exists(progress_file):
                    os.remove(progress_file)
            except Exception as e:
                logger.warning(f"Could not remove progress file: {e}")
        
        return index_dir, total_docs_new
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
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
    Get statistics about the search index (across all shards).
    
    Args:
        index_dir: Directory containing the index. If None, uses get_index_dir()
    
    Returns:
        Dictionary with index statistics
    """
    if index_dir is None:
        index_dir = get_index_dir()
    
    shards_dir = os.path.join(index_dir, "shards")
    
    if not os.path.exists(shards_dir):
        raise FileNotFoundError(f"Shards directory does not exist: {shards_dir}")
    
    try:
        # Find all shards
        shard_names = sorted([d for d in os.listdir(shards_dir) if d.startswith("shard_")])
        
        if len(shard_names) == 0:
            raise FileNotFoundError(f"No shards found in {shards_dir}")
        
        # Collect stats from all shards
        total_docs = 0
        total_length = 0.0
        
        for shard_name in shard_names:
            shard_path = os.path.join(shards_dir, shard_name)
            shard_db = xapian.Database(shard_path)
            
            shard_docs = shard_db.get_doccount()
            total_docs += shard_docs
            total_length += shard_db.get_avlength() * shard_docs
            
            shard_db.close()
        
        avg_doc_length = total_length / total_docs if total_docs > 0 else 0.0
        
        stats = {
            'index_dir': index_dir,
            'num_shards': len(shard_names),
            'num_documents': total_docs,
            'avg_doc_length': avg_doc_length,
        }
        
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
    
    # Build arguments
    parser.add_argument("--no-resume", action="store_true",
                        help="Start indexing from scratch (default: resume from previous progress)")
    parser.add_argument("--no-sync", action="store_true",
                        help="Disable fsync for faster indexing (RISKY: data loss if crash)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of parquet files to process (default: all files)")
    
    # Search arguments
    parser.add_argument("-n", "--max-results", type=int, default=10, 
                        help="Maximum number of search results (default: 10)")
    parser.add_argument("--slop", type=int, default=20,
                        help="Phrase matching flexibility - extra distance allowed between terms (default: 20)")
    
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
            index_dir=args.index_dir,
            resume=not args.no_resume,
            no_sync=args.no_sync,
            num_workers=args.workers,
            max_files=args.max_files
        )
        logger.info("="*80)
        logger.info(f"Index built successfully with {total_docs} documents")
        logger.info(f"Index location: {index_dir}")
        
    elif args.search:
        logger.info("="*80)
        logger.info("Searching index")
        logger.info("="*80)
        
        # Use SearchContext with search_phrase for robust phrase matching
        with SearchContext(index_dir=args.index_dir, data_dir=args.data_dir) as ctx:
            logger.info(f"Index contains {ctx.num_docs:,} documents")
            
            results = ctx.search_phrase(
                phrase_text=args.search,
                max_results=args.max_results,
                slop=args.slop
            )
        
        logger.info("="*80)
        logger.info(f"Search Results ({len(results)} matches)")
        logger.info("="*80)
        
        for result in results:
            print(f"\nRank #{result['rank']} - Score: {result['score']:.1%}")
            print(f"Location: file {result['file_idx']}, row_group {result['rg_idx']}, doc {result['doc_idx']}")
            
            # Show full text
            text = result['text']
            print(f"Text:\n{text}")
            print("-" * 80)
        
        if len(results) == 0:
            logger.warning("No results found. Try different search terms.")
    
    elif args.stats:
        logger.info("="*80)
        logger.info("Index Statistics")
        logger.info("="*80)
        stats = get_index_stats(index_dir=args.index_dir)
        
        print(f"\nIndex Directory: {stats['index_dir']}")
        print(f"Number of Shards: {stats['num_shards']}")
        print(f"Total Documents: {stats['num_documents']:,}")
        print(f"Average Document Length: {stats['avg_doc_length']:.1f} terms")

