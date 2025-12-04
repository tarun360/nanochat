"""
Script to inspect contamination matches by retrieving full documents.

Reads contamination_report.json and displays the full text of matched documents
from the parquet files to help determine if matches are real contamination.

Usage:
    python -m data_analysis.inspect_matches contamination_results/contamination_report.json
    python -m data_analysis.inspect_matches contamination_results/contamination_report.json --match-type q_strong
    python -m data_analysis.inspect_matches contamination_results/contamination_report.json --dataset gsm8k --split test
"""

import json
import argparse
from datasets import load_dataset
from data_analysis.search_index import retrieve_text_from_parquet

def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)

# Cache for loaded datasets to avoid reloading
_dataset_cache = {}

def load_dataset_example(dataset_name, split_name, example_idx):
    """Load a specific example from a dataset (with caching)."""
    cache_key = (dataset_name, split_name)
    
    # Check cache first
    if cache_key not in _dataset_cache:
        if dataset_name == 'GSM8K':
            _dataset_cache[cache_key] = load_dataset("openai/gsm8k", "main", split=split_name)
        elif dataset_name.startswith('MATH/'):
            subject = dataset_name.split('/')[1]
            _dataset_cache[cache_key] = load_dataset("EleutherAI/hendrycks_math", subject, split=split_name)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get example from cached dataset
    dataset = _dataset_cache[cache_key]
    example = dataset[example_idx]
    if dataset_name == 'GSM8K':
        return example.get('question', ''), example.get('answer', '')
    elif dataset_name.startswith('MATH/'):
        return example.get('problem', ''), example.get('solution', '')
    
    raise ValueError(f"Unknown dataset: {dataset_name}")

def print_match_details(match_info, dataset_name, split_name, example_idx, data_dir=None, match_type=None):
    """Print details about a single contamination match.
    
    Args:
        match_info: Match information dictionary
        dataset_name: Name of the dataset
        split_name: Name of the split
        example_idx: Index of the example
        data_dir: Data directory for parquet files
        match_type: Type of match being displayed (e.g., 'q_strong', 'a_strong') - only show that type
    """
    
    print_separator('=')
    print(f"MATCH: {dataset_name} {split_name} - Example #{example_idx}")
    print_separator('=')
    
    # Load and display the full question and answer from the dataset
    question, answer = load_dataset_example(dataset_name, split_name, example_idx)
    print(f"\nFull Question from Dataset:")
    print_separator('-')
    print(question)
    print_separator('-')
    
    print(f"\nFull Answer from Dataset:")
    print_separator('-')
    print(answer)
    print_separator('-')
    
    # Determine which matches to show based on match_type filter
    show_q_match = match_info['q_match'] and (not match_type or match_type.startswith('q'))
    show_a_match = match_info['a_match'] and (not match_type or match_type.startswith('a'))
    
    # Display Question match if exists and should be shown
    if show_q_match:
        q_match = match_info['q_match']
        print(f"\nQuestion Match Score: {q_match['score']:.1%}")
        print(f"Location: file={q_match['file_idx']}, rg={q_match['rg_idx']}, doc={q_match['doc_idx']}")
        
        # Display n-gram details if present
        if 'ngram_details' in q_match:
            print(f"\nN-gram Match Details:")
            for ngram in q_match['ngram_details']:
                status = "✓ MATCHED" if ngram['matched'] else "✗ No match"
                print(f"  {ngram['position'].upper():8} - {status:12} - Score: {ngram['score']:.1%} ({ngram['match_count']} matches)")
                # Only show n-gram terms if they're available (for short phrases)
                if ngram['matched'] and 'ngram_terms' in ngram and ngram['ngram_terms']:
                    print(f"    Matched n-gram: {ngram['ngram_terms']}")
        
        print(f"\nRetrieving full document from parquet...")
        
        full_text = retrieve_text_from_parquet(
            q_match['file_idx'],
            q_match['rg_idx'],
            q_match['doc_idx'],
            data_dir
        )
        
        if full_text:
            print(f"\nFull Matched Document (Question search):")
            print_separator('-')
            print(full_text)
            print_separator('-')
            print(f"Document length: {len(full_text)} characters")
        else:
            print("ERROR: Could not retrieve document")
    
    # Display Answer match if exists and should be shown
    # Also show if it's different from Question match (when showing both)
    if show_a_match and (not show_q_match or 
                         match_info['a_match']['file_idx'] != match_info.get('q_match', {}).get('file_idx') or
                         match_info['a_match']['rg_idx'] != match_info.get('q_match', {}).get('rg_idx') or
                         match_info['a_match']['doc_idx'] != match_info.get('q_match', {}).get('doc_idx')):
        a_match = match_info['a_match']
        print(f"\nAnswer Match Score: {a_match['score']:.1%}")
        print(f"Location: file={a_match['file_idx']}, rg={a_match['rg_idx']}, doc={a_match['doc_idx']}")
        
        # Display n-gram details if present
        if 'ngram_details' in a_match:
            print(f"\nN-gram Match Details:")
            for ngram in a_match['ngram_details']:
                status = "✓ MATCHED" if ngram['matched'] else "✗ No match"
                print(f"  {ngram['position'].upper():8} - {status:12} - Score: {ngram['score']:.1%} ({ngram['match_count']} matches)")
                # Only show n-gram terms if they're available (for short phrases)
                if ngram['matched'] and 'ngram_terms' in ngram and ngram['ngram_terms']:
                    print(f"    Matched n-gram: {ngram['ngram_terms']}")
        
        print(f"\nRetrieving full document from parquet...")
        
        full_text = retrieve_text_from_parquet(
            a_match['file_idx'],
            a_match['rg_idx'],
            a_match['doc_idx'],
            data_dir
        )
        
        if full_text:
            print(f"\nFull Matched Document (Answer search):")
            print_separator('-')
            print(full_text)
            print_separator('-')
            print(f"Document length: {len(full_text)} characters")
        else:
            print("ERROR: Could not retrieve document")
    
    print("\n")

def main():
    parser = argparse.ArgumentParser(
        description="Inspect contamination matches by retrieving full documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all strong Question matches
  python -m data_analysis.inspect_matches contamination_results/contamination_report.json
  
  # Show specific dataset
  python -m data_analysis.inspect_matches contamination_results/contamination_report.json --dataset gsm8k
  
  # Show specific split
  python -m data_analysis.inspect_matches contamination_results/contamination_report.json --dataset gsm8k --split test
  
  # Show all Answer strong matches
  python -m data_analysis.inspect_matches contamination_results/contamination_report.json --match-type a_strong
        """
    )
    
    parser.add_argument("json_file", help="Path to contamination_report.json")
    parser.add_argument("--dataset", choices=['gsm8k', 'math'], help="Filter by dataset")
    parser.add_argument("--subject", help="Filter by MATH subject (only with --dataset math)")
    parser.add_argument("--split", choices=['train', 'test'], help="Filter by split")
    parser.add_argument("--match-type", choices=['q_strong', 'q_moderate', 'a_strong', 'a_moderate'],
                        default='q_strong',
                        help="Type of matches to show (default: q_strong)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of matches to display")
    parser.add_argument("--data-dir", type=str, help="Data directory for parquet files")
    
    args = parser.parse_args()
    
    # Load the JSON report
    print(f"Loading contamination report from: {args.json_file}")
    with open(args.json_file, 'r') as f:
        report = json.load(f)
    
    print(f"Report generated: {report['metadata']['generated_at']}")
    print(f"Total docs in index: {report['metadata']['total_docs_in_index']:,}")
    print()
    
    matches_to_display = []
    
    # Collect matches based on filters
    if not args.dataset or args.dataset == 'gsm8k':
        gsm8k_data = report.get('gsm8k', {})
        for split_name in ['train', 'test']:
            if args.split and args.split != split_name:
                continue
            
            split_data = gsm8k_data.get(split_name, {})
            detailed_matches = split_data.get('detailed_matches', [])
            
            # Filter by match type
            for match in detailed_matches:
                # Check if this match has the type we're looking for
                if args.match_type.startswith('q'):
                    if match['q_category'] == args.match_type.split('_')[1]:
                        matches_to_display.append(('GSM8K', split_name, match))
                elif args.match_type.startswith('a'):
                    if match['a_category'] == args.match_type.split('_')[1]:
                        matches_to_display.append(('GSM8K', split_name, match))
    
    if not args.dataset or args.dataset == 'math':
        math_data = report.get('math', {})
        for subject, subject_data in math_data.items():
            if args.subject and args.subject != subject:
                continue
            
            for split_name in ['train', 'test']:
                if args.split and args.split != split_name:
                    continue
                
                split_data = subject_data.get(split_name, {})
                detailed_matches = split_data.get('detailed_matches', [])
                
                # Filter by match type
                for match in detailed_matches:
                    if args.match_type.startswith('q'):
                        if match['q_category'] == args.match_type.split('_')[1]:
                            matches_to_display.append((f'MATH/{subject}', split_name, match))
                    elif args.match_type.startswith('a'):
                        if match['a_category'] == args.match_type.split('_')[1]:
                            matches_to_display.append((f'MATH/{subject}', split_name, match))
    
    # Apply limit
    if args.limit:
        matches_to_display = matches_to_display[:args.limit]
    
    # Display matches
    print_separator('=')
    print(f"FOUND {len(matches_to_display)} MATCHES TO DISPLAY")
    print(f"Match type: {args.match_type}")
    print_separator('=')
    print()
    
    if len(matches_to_display) == 0:
        print("No matches found with the specified filters.")
        return
    
    for idx, (dataset_name, split_name, match) in enumerate(matches_to_display, 1):
        print(f"\n{'='*80}")
        print(f"MATCH {idx}/{len(matches_to_display)}")
        print_match_details(match, dataset_name, split_name, match['example_idx'], args.data_dir, match_type=args.match_type)
        
        if idx < len(matches_to_display):
            response = input("Press Enter for next match, or 'q' to quit: ")
            if response.lower() == 'q':
                break
    
    print(f"\nDisplayed {min(idx, len(matches_to_display))} of {len(matches_to_display)} matches.")

if __name__ == "__main__":
    main()

