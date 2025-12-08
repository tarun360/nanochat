#!/usr/bin/env python3
"""
Test script to verify if OR queries affect scoring in n-gram search.
Compares scores between:
1. Single n-gram query
2. Same query as part of an OR query with other n-grams
"""

import sys
import xapian
from data_analysis.search_index import SearchContext

def test_single_vs_or_scoring(ctx, phrase, ngram_size=10):
    """
    Test if a single n-gram query has the same score as when it's part of an OR query.
    """
    print(f"\n{'='*80}")
    print(f"Testing scoring: '{phrase[:100]}...'")
    print('='*80)
    
    # Use TermGenerator to get terms (same as search_phrase_ngrams does)
    termgen = xapian.TermGenerator()
    termgen.set_stemmer(xapian.Stem("en"))
    
    # Create a temporary document to extract terms
    temp_doc = xapian.Document()
    termgen.set_document(temp_doc)
    termgen.index_text(phrase)
    
    # Extract terms with positions
    pos_to_terms = {}
    for titem in temp_doc.termlist():
        termname = titem.term
        pos_iter = titem.positer
        for pos in pos_iter:
            pos_to_terms.setdefault(pos, []).append(termname)
    
    # Build term sequence
    sequence = []
    for pos in sorted(pos_to_terms.keys()):
        sequence.extend(pos_to_terms[pos])
    
    if len(sequence) <= ngram_size:
        print(f"⚠️  Phrase too short ({len(sequence)} terms), need at least {ngram_size + 1} terms")
        return
    
    # Test 1: Single n-gram query (first n-gram)
    first_ngram = sequence[:ngram_size]
    if len(first_ngram) == 1:
        single_query = xapian.Query(first_ngram[0])
    else:
        window = len(first_ngram) + 0  # slop=0
        single_query = xapian.Query(xapian.Query.OP_PHRASE, first_ngram, window)
    
    enquire_single = xapian.Enquire(ctx.database)
    enquire_single.set_query(single_query)
    matches_single = enquire_single.get_mset(0, 1)
    
    single_score = 0.0
    single_docid = None
    if matches_single.size() > 0:
        match = next(iter(matches_single))
        single_score = match.percent / 100.0
        single_docid = match.docid
    
    print(f"\n1. Single n-gram query (first {ngram_size} terms):")
    print(f"   Score: {single_score:.4f} ({single_score:.1%})")
    print(f"   Doc ID: {single_docid}")
    
    # Test 2: MAX query with multiple n-grams (including the same first n-gram)
    all_ngram_queries = []
    for i in range(len(sequence) - ngram_size + 1):
        term_subsequence = sequence[i : i + ngram_size]
        if len(term_subsequence) == 1:
            all_ngram_queries.append(xapian.Query(term_subsequence[0]))
        else:
            window = len(term_subsequence) + 0  # slop=0
            all_ngram_queries.append(xapian.Query(xapian.Query.OP_PHRASE, term_subsequence, window))
    
    # Create MAX query (OP_MAX takes the maximum weight from any matching subquery)
    if len(all_ngram_queries) == 1:
        max_query = all_ngram_queries[0]
    else:
        max_query = xapian.Query(xapian.Query.OP_MAX, all_ngram_queries)
    
    enquire_max = xapian.Enquire(ctx.database)
    enquire_max.set_query(max_query)
    matches_max = enquire_max.get_mset(0, 1)
    
    max_score = 0.0
    max_docid = None
    if matches_max.size() > 0:
        match = next(iter(matches_max))
        max_score = match.percent / 100.0
        max_docid = match.docid
    
    print(f"\n2. MAX query with {len(all_ngram_queries)} n-grams (including the same first n-gram):")
    print(f"   Score: {max_score:.4f} ({max_score:.1%})")
    print(f"   Doc ID: {max_docid}")
    
    # Compare
    print(f"\n{'='*80}")
    if single_docid == max_docid and single_docid is not None:
        print(f"✓ Same document matched in both cases")
        score_diff = abs(single_score - max_score)
        score_diff_pct = (score_diff / single_score * 100) if single_score > 0 else 0
        print(f"  Score difference: {score_diff:.6f} ({score_diff_pct:.2f}%)")
        
        if score_diff < 0.0001:
            print(f"  ✓ Scores are essentially identical (difference < 0.01%)")
            print(f"  ✓ OP_MAX correctly preserves the maximum score!")
        elif max_score < single_score:
            print(f"  ⚠️  MAX query has LOWER score than single query!")
            print(f"     This is unexpected - OP_MAX should preserve the maximum score.")
        else:
            print(f"  ℹ️  MAX query has HIGHER score (might match multiple n-grams)")
    else:
        print(f"⚠️  Different documents matched!")
        print(f"   Single query matched doc {single_docid}")
        print(f"   MAX query matched doc {max_docid}")
    
    return single_score, max_score, single_docid == max_docid


def main():
    print("="*80)
    print("N-GRAM OR QUERY SCORING TEST")
    print("="*80)
    print("\nThis script tests if using MAX queries preserves scoring compared to single queries.")
    print("We compare:")
    print("1. A single n-gram query")
    print("2. The same query as part of an OP_MAX query with other n-grams")
    print("\nOP_MAX should preserve the maximum score from any matching subquery.")
    print()
    
    with SearchContext() as ctx:
        print(f"Index loaded: {ctx.num_docs:,} documents")
        
        # Test with a known phrase that exists in the index
        test_phrases = [
            "So why not do the same for Researchers at the Swiss Federal Institute of Technology",
        ]
        
        all_results = []
        for phrase in test_phrases:
            result = test_single_vs_or_scoring(ctx, phrase, ngram_size=10)
            if result:
                all_results.append(result)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if all_results:
            same_doc_count = sum(1 for _, _, same_doc in all_results if same_doc)
            print(f"Tests run: {len(all_results)}")
            print(f"Same document matched: {same_doc_count}/{len(all_results)}")
            
            score_diffs = []
            for single_score, max_score, same_doc in all_results:
                if same_doc and single_score > 0:
                    diff_pct = abs(single_score - max_score) / single_score * 100
                    score_diffs.append(diff_pct)
            
            if score_diffs:
                avg_diff = sum(score_diffs) / len(score_diffs)
                max_diff = max(score_diffs)
                print(f"\nAverage score difference: {avg_diff:.2f}%")
                print(f"Maximum score difference: {max_diff:.2f}%")
                
                if avg_diff < 1.0:
                    print(f"\n✓ Scores are very similar - OP_MAX correctly preserves maximum scores!")
                elif max_diff > 5.0:
                    print(f"\n⚠️  Significant score differences detected - OP_MAX may not be working as expected")
                else:
                    print(f"\nℹ️  Minor score differences - likely within normal variation")
        else:
            print("No valid test results")
        
        return 0


if __name__ == "__main__":
    sys.exit(main())

