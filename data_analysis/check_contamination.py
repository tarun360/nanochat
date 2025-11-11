"""
Dataset contamination checker for nanochat.

This script checks if evaluation datasets (GSM8K, MATH) appear in the training data
by searching the indexed parquet files.

Contamination levels:
- Strong match (≥90%): High confidence of contamination
- Moderate match (50-89%): Possible contamination, needs review
- Weak match (<50%): Likely not contamination

Usage:
    python -m data_analysis.check_contamination --all
    python -m data_analysis.check_contamination --dataset gsm8k
    python -m data_analysis.check_contamination --dataset math --subject algebra
"""

import os
import argparse
import logging
import json
import time
from datetime import datetime
from datasets import load_dataset

from data_analysis.search_index import search, get_index_stats, get_index_dir

# Setup logging
logger = logging.getLogger(__name__)

# MATH dataset subjects
MATH_SUBJECTS = [
    'algebra',
    'counting_and_probability', 
    'geometry',
    'intermediate_algebra',
    'number_theory',
    'prealgebra',
    'precalculus'
]

# -----------------------------------------------------------------------------
# Core contamination checking functions

def categorize_score(score):
    """Categorize a match score into strong/moderate/weak."""
    if score >= 0.90:
        return 'strong'
    elif score >= 0.50:
        return 'moderate'
    else:
        return 'weak'


def check_single_example(example_idx, question, answer, index_dir=None):
    """
    Check a single example for contamination.
    
    Args:
        example_idx: Index of the example in the dataset
        question: Question text
        answer: Answer text
        index_dir: Path to search index
    
    Returns:
        Dictionary with contamination results
    """
    result = {
        'example_idx': example_idx,
        'question_preview': question[:100],
        'qa_score': 0.0,
        'qa_category': 'weak',
        'qa_match': None,
        'q_score': 0.0,
        'q_category': 'weak',
        'q_match': None,
    }
    
    try:
        # Search Question + Answer combined
        qa_query = question + " " + answer
        qa_matches = search(qa_query, index_dir=index_dir, max_results=1, fuzzy=True)
        
        if qa_matches and len(qa_matches) > 0:
            result['qa_score'] = qa_matches[0]['score']
            result['qa_category'] = categorize_score(qa_matches[0]['score'])
            result['qa_match'] = {
                'score': qa_matches[0]['score'],
                'file_idx': qa_matches[0]['file_idx'],
                'rg_idx': qa_matches[0]['rg_idx'],
                'doc_idx': qa_matches[0]['doc_idx'],
                'text_preview': qa_matches[0]['text'][:200] if qa_matches[0]['text'] else None
            }
        
        # Search Question only
        q_matches = search(question, index_dir=index_dir, max_results=1, fuzzy=True)
        
        if q_matches and len(q_matches) > 0:
            result['q_score'] = q_matches[0]['score']
            result['q_category'] = categorize_score(q_matches[0]['score'])
            result['q_match'] = {
                'score': q_matches[0]['score'],
                'file_idx': q_matches[0]['file_idx'],
                'rg_idx': q_matches[0]['rg_idx'],
                'doc_idx': q_matches[0]['doc_idx'],
                'text_preview': q_matches[0]['text'][:200] if q_matches[0]['text'] else None
            }
    
    except Exception as e:
        logger.error(f"Error checking example {example_idx}: {e}")
    
    return result


def check_dataset_split(dataset_name, split_name, dataset, index_dir=None):
    """
    Check an entire dataset split for contamination.
    
    Args:
        dataset_name: Name of the dataset (e.g., "GSM8K", "MATH/algebra")
        split_name: Split name (e.g., "train", "test")
        dataset: HuggingFace dataset object
        index_dir: Path to search index
    
    Returns:
        Dictionary with aggregated results
    """
    logger.info(f"Checking {dataset_name} {split_name} split ({len(dataset)} examples)...")
    
    results = {
        'dataset_name': dataset_name,
        'split_name': split_name,
        'total_examples': len(dataset),
        'qa_strong': [],
        'qa_moderate': [],
        'q_strong': [],
        'q_moderate': [],
        'detailed_matches': []
    }
    
    start_time = time.time()
    
    for idx in range(len(dataset)):
        example = dataset[idx]
        question = example['question']
        answer = example['answer']
        
        # Check this example
        check_result = check_single_example(idx, question, answer, index_dir)
        
        # Categorize based on Q+A score
        if check_result['qa_category'] == 'strong':
            results['qa_strong'].append(idx)
            results['detailed_matches'].append(check_result)
        elif check_result['qa_category'] == 'moderate':
            results['qa_moderate'].append(idx)
            results['detailed_matches'].append(check_result)
        
        # Categorize based on Q-only score
        if check_result['q_category'] == 'strong':
            results['q_strong'].append(idx)
            if check_result['qa_category'] == 'weak':  # Only add if not already in detailed
                results['detailed_matches'].append(check_result)
        elif check_result['q_category'] == 'moderate':
            results['q_moderate'].append(idx)
        
        # Log progress every 100 examples
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  Checked {idx + 1}/{len(dataset)} examples ({rate:.1f} ex/sec)")
    
    elapsed = time.time() - start_time
    logger.info(f"  Completed {dataset_name} {split_name}: {len(dataset)} examples in {elapsed:.1f}s")
    
    return results


# -----------------------------------------------------------------------------
# Dataset-specific processing functions

def process_gsm8k(index_dir=None):
    """Process GSM8K dataset (both train and test splits)."""
    logger.info("=" * 80)
    logger.info("Processing GSM8K Dataset")
    logger.info("=" * 80)
    
    # Load dataset
    logger.info("Loading GSM8K dataset from HuggingFace...")
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    logger.info(f"Loaded: {len(train_dataset)} train, {len(test_dataset)} test examples")
    
    # Check both splits
    train_results = check_dataset_split("GSM8K", "train", train_dataset, index_dir)
    test_results = check_dataset_split("GSM8K", "test", test_dataset, index_dir)
    
    return {
        'train': train_results,
        'test': test_results
    }


def process_math_subject(subject, index_dir=None):
    """Process one subject of MATH dataset (both train and test splits)."""
    logger.info(f"Loading MATH/{subject} dataset from HuggingFace...")
    
    train_dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
    test_dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
    
    logger.info(f"Loaded MATH/{subject}: {len(train_dataset)} train, {len(test_dataset)} test examples")
    
    # Check both splits
    train_results = check_dataset_split(f"MATH/{subject}", "train", train_dataset, index_dir)
    test_results = check_dataset_split(f"MATH/{subject}", "test", test_dataset, index_dir)
    
    return {
        'train': train_results,
        'test': test_results
    }


def process_math_all(index_dir=None):
    """Process all MATH dataset subjects."""
    logger.info("=" * 80)
    logger.info("Processing MATH Dataset (all subjects)")
    logger.info("=" * 80)
    
    results = {}
    for subject in MATH_SUBJECTS:
        logger.info(f"\nProcessing subject: {subject}")
        logger.info("-" * 80)
        subject_results = process_math_subject(subject, index_dir)
        if subject_results:
            results[subject] = subject_results
    
    return results


# -----------------------------------------------------------------------------
# Report generation functions

def generate_text_report(gsm8k_results, math_results, output_file):
    """Generate human-readable text report."""
    
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("DATASET CONTAMINATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Index info
        try:
            stats = get_index_stats()
            f.write(f"Index: {stats['index_dir']}\n")
            f.write(f"Total documents in index: {stats['num_documents']:,}\n")
        except:
            f.write("Index: (stats unavailable)\n")
        
        f.write("\n\n")
        
        # GSM8K Section
        f.write("GSM8K DATASET\n")
        f.write("=" * 80 + "\n\n")
        
        for split in ['test', 'train']:
            split_data = gsm8k_results[split]
            total = split_data['total_examples']
            
            f.write(f"{split.upper()} Split ({total:,} examples):\n")
            f.write(f"\n  Question + Answer:\n")
            f.write(f"    ├─ Strong matches (≥90%):    {len(split_data['qa_strong'])} ({len(split_data['qa_strong'])/total*100:.2f}%)\n")
            f.write(f"    └─ Moderate matches (50-89%): {len(split_data['qa_moderate'])} ({len(split_data['qa_moderate'])/total*100:.2f}%)\n")
            f.write(f"\n  Question Only:\n")
            f.write(f"    ├─ Strong matches (≥90%):    {len(split_data['q_strong'])} ({len(split_data['q_strong'])/total*100:.2f}%)\n")
            f.write(f"    └─ Moderate matches (50-89%): {len(split_data['q_moderate'])} ({len(split_data['q_moderate'])/total*100:.2f}%)\n")
            f.write("\n")
        
        # MATH Section
        f.write("\n" + "=" * 80 + "\n")
        f.write("MATH DATASET\n")
        f.write("=" * 80 + "\n\n")
        
        for subject in MATH_SUBJECTS:
            if subject not in math_results:
                continue
            
            f.write(f"\n{subject.upper().replace('_', ' ')}\n")
            f.write("-" * 80 + "\n")
            
            for split in ['test', 'train']:
                split_data = math_results[subject][split]
                total = split_data['total_examples']
                
                f.write(f"\n  {split.upper()} Split ({total:,} examples):\n")
                f.write(f"    Question + Answer:\n")
                f.write(f"      ├─ Strong matches (≥90%):    {len(split_data['qa_strong'])} ({len(split_data['qa_strong'])/total*100:.2f}%)\n")
                f.write(f"      └─ Moderate matches (50-89%): {len(split_data['qa_moderate'])} ({len(split_data['qa_moderate'])/total*100:.2f}%)\n")
                f.write(f"    Question Only:\n")
                f.write(f"      ├─ Strong matches (≥90%):    {len(split_data['q_strong'])} ({len(split_data['q_strong'])/total*100:.2f}%)\n")
                f.write(f"      └─ Moderate matches (50-89%): {len(split_data['q_moderate'])} ({len(split_data['q_moderate'])/total*100:.2f}%)\n")
        
        # Summary
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("SUMMARY - Most Contaminated (by strong Q+A matches)\n")
        f.write("=" * 80 + "\n\n")
        
        contamination_list = []
        
        # Collect GSM8K
        for split in ['test', 'train']:
            split_data = gsm8k_results[split]
            pct = len(split_data['qa_strong']) / split_data['total_examples'] * 100
            contamination_list.append((f"GSM8K/{split}", pct, len(split_data['qa_strong'])))
        
        # Collect MATH
        for subject in MATH_SUBJECTS:
            if subject not in math_results:
                continue
            for split in ['test', 'train']:
                split_data = math_results[subject][split]
                pct = len(split_data['qa_strong']) / split_data['total_examples'] * 100
                contamination_list.append((f"MATH/{subject}/{split}", pct, len(split_data['qa_strong'])))
        
        # Sort by contamination percentage
        contamination_list.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (name, pct, count) in enumerate(contamination_list[:10], 1):
            f.write(f"  {rank:2d}. {name:40s} {pct:6.2f}% ({count} matches)\n")
    
    logger.info(f"Text report saved to: {output_file}")


def save_json_report(gsm8k_results, math_results, output_file, metadata):
    """Save detailed JSON report."""
    
    report = {
        'metadata': metadata,
        'gsm8k': gsm8k_results,
        'math': math_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"JSON report saved to: {output_file}")


def save_detailed_matches(gsm8k_results, math_results, output_dir):
    """Save detailed match information to separate JSON files."""
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving detailed matches to: {output_dir}")
    
    # Save GSM8K matches
    for split in ['train', 'test']:
        matches = gsm8k_results[split]['detailed_matches']
        if matches:
            filepath = os.path.join(output_dir, f'gsm8k_{split}.json')
            with open(filepath, 'w') as f:
                json.dump(matches, f, indent=2)
            logger.info(f"  Saved {len(matches)} GSM8K {split} matches")
    
    # Save MATH matches
    for subject in MATH_SUBJECTS:
        if subject not in math_results:
            continue
        for split in ['train', 'test']:
            matches = math_results[subject][split]['detailed_matches']
            if matches:
                filepath = os.path.join(output_dir, f'math_{subject}_{split}.json')
                with open(filepath, 'w') as f:
                    json.dump(matches, f, indent=2)
                logger.info(f"  Saved {len(matches)} MATH/{subject} {split} matches")


# -----------------------------------------------------------------------------
# Main execution

def main():
    parser = argparse.ArgumentParser(
        description="Check dataset contamination in training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all datasets
  python -m data_analysis.check_contamination --all
  
  # Check only GSM8K
  python -m data_analysis.check_contamination --dataset gsm8k
  
  # Check only MATH dataset (all subjects)
  python -m data_analysis.check_contamination --dataset math
  
  # Check specific MATH subject
  python -m data_analysis.check_contamination --dataset math --subject algebra
  
  # Custom output location
  python -m data_analysis.check_contamination --all --output-dir ./contamination_results
        """
    )
    
    # What to check
    parser.add_argument("--all", action="store_true", help="Check all datasets")
    parser.add_argument("--dataset", type=str, choices=['gsm8k', 'math'], 
                        help="Specific dataset to check")
    parser.add_argument("--subject", type=str, choices=MATH_SUBJECTS,
                        help="Specific MATH subject (only valid with --dataset math)")
    
    # Configuration
    parser.add_argument("--index-dir", type=str, help="Search index directory")
    parser.add_argument("--output-dir", type=str, default="./contamination_results",
                        help="Output directory for reports (default: ./contamination_results)")
    parser.add_argument("--save-detailed", action="store_true",
                        help="Save detailed match information to JSON files")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.dataset:
        parser.error("Must specify --all or --dataset")
    
    if args.subject and args.dataset != 'math':
        parser.error("--subject can only be used with --dataset math")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify index exists
    index_dir = args.index_dir if args.index_dir else get_index_dir()
    logger.info(f"Using search index at: {index_dir}")
    
    try:
        stats = get_index_stats(index_dir)
        logger.info(f"Index contains {stats['num_documents']:,} documents")
    except Exception as e:
        logger.error(f"Failed to access index: {e}")
        logger.error("Please build the index first!")
        return
    
    # Prepare metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'index_path': index_dir,
        'total_docs_in_index': stats['num_documents'],
        'thresholds': {
            'strong': 0.90,
            'moderate': 0.50
        }
    }
    
    gsm8k_results = None
    math_results = None
    
    start_time = time.time()
    
    # Process datasets based on arguments
    if args.all or args.dataset == 'gsm8k':
        gsm8k_results = process_gsm8k(index_dir)
    
    if args.all or args.dataset == 'math':
        if args.subject:
            # Process only one subject
            logger.info("=" * 80)
            logger.info(f"Processing MATH Dataset - {args.subject}")
            logger.info("=" * 80)
            subject_results = process_math_subject(args.subject, index_dir)
            math_results = {args.subject: subject_results} if subject_results else {}
        else:
            # Process all subjects
            math_results = process_math_all(index_dir)
    
    total_elapsed = time.time() - start_time
    
    # Generate reports
    logger.info("\n" + "=" * 80)
    logger.info("Generating Reports")
    logger.info("=" * 80)
    
    # Text report
    if gsm8k_results and math_results:
        text_report_file = os.path.join(args.output_dir, "contamination_report.txt")
        generate_text_report(gsm8k_results, math_results, text_report_file)
    
    # JSON report
    json_report_file = os.path.join(args.output_dir, "contamination_report.json")
    save_json_report(
        gsm8k_results if gsm8k_results else {},
        math_results if math_results else {},
        json_report_file,
        metadata
    )
    
    # Detailed matches
    if args.save_detailed:
        matches_dir = os.path.join(args.output_dir, "detailed_matches")
        save_detailed_matches(
            gsm8k_results if gsm8k_results else {},
            math_results if math_results else {},
            matches_dir
        )
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Contamination Check Complete!")
    logger.info("=" * 80)
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info(f"Reports saved to: {args.output_dir}")
    logger.info(f"  - Text report: contamination_report.txt")
    logger.info(f"  - JSON report: contamination_report.json")
    if args.save_detailed:
        logger.info(f"  - Detailed matches: detailed_matches/")
    
    logger.info("\nOpen contamination_report.txt to view results!")


if __name__ == "__main__":
    main()

