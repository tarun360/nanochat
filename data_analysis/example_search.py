"""
Example script demonstrating how to use the search index.
This shows both building an index and searching it programmatically.
"""

import logging
from search_index import build_index, search, get_index_stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Example usage of the search index."""
    
    # Option 1: Build the index (only needs to be done once)
    # Uncomment to build:
    # logger.info("Building search index...")
    # index_dir, num_docs = build_index()
    # logger.info(f"Built index with {num_docs} documents at {index_dir}")
    
    # Option 2: Get index statistics
    logger.info("\nGetting index statistics...")
    try:
        stats = get_index_stats()
        logger.info(f"Index has {stats['num_documents']} documents")
        logger.info(f"Average document length: {stats['avg_doc_length']:.1f} terms")
    except FileNotFoundError:
        logger.error("Index not found. Please build it first with --build")
        return
    
    # Option 3: Search the index
    logger.info("\nSearching the index...")
    query = "machine learning and artificial intelligence"
    results = search(query, max_results=5, fuzzy=True)
    
    logger.info(f"\nFound {len(results)} results for '{query}':\n")
    for result in results:
        print(f"Rank {result['rank']}: Score={result['score']:.1%}")
        print(f"Location: (file={result['file_idx']}, rg={result['rg_idx']}, doc={result['doc_idx']})")
        
        # Show first 150 characters
        preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
        print(f"Preview: {preview}")
        print("-" * 80)

if __name__ == "__main__":
    main()

