#!/usr/bin/env python3
"""
Test script to verify phrase search is working correctly.
Searches for known phrases from documents that appeared in false positive results.
"""

import sys
from data_analysis.search_index import SearchContext

def test_phrase(ctx, phrase, expected_in_result=True):
    """Test a single phrase search."""
    print(f"\n{'='*80}")
    print(f"Testing phrase: '{phrase[:80]}...'")
    print('='*80)
    
    results = ctx.search_phrase(phrase, max_results=3, slop=0)
    
    if not results:
        print("❌ NO RESULTS FOUND")
        if expected_in_result:
            print("   ERROR: Expected to find this phrase!")
        else:
            print("   ✓ Correctly found no matches")
        return not expected_in_result
    
    print(f"✓ Found {len(results)} results")
    
    # Check if phrase actually appears in results
    phrase_lower = phrase.lower()
    found_exact = False
    
    for i, result in enumerate(results, 1):
        text_lower = result['text'].lower() if result['text'] else ""
        contains_phrase = phrase_lower in text_lower
        
        print(f"\nResult {i}:")
        print(f"  Score: {result['score']:.1%}")
        print(f"  Location: file={result['file_idx']}, rg={result['rg_idx']}, doc={result['doc_idx']}")
        print(f"  Contains exact phrase: {'✓ YES' if contains_phrase else '❌ NO'}")
        
        if contains_phrase:
            found_exact = True
            # Show context around the phrase
            idx = text_lower.find(phrase_lower)
            start = max(0, idx - 50)
            end = min(len(text_lower), idx + len(phrase_lower) + 50)
            context = result['text'][start:end]
            print(f"  Context: ...{context}...")
    
    if expected_in_result and not found_exact:
        print("\n❌ ERROR: Phrase not found in any result!")
        return False
    elif not expected_in_result and found_exact:
        print("\n❌ ERROR: Phrase found but should not have been!")
        return False
    else:
        print("\n✓ Test passed!")
        return True


def main():
    print("="*80)
    print("PHRASE SEARCH VERIFICATION TEST")
    print("="*80)
    print("\nThis script tests that phrase search works correctly by:")
    print("1. Searching for known phrases from indexed documents")
    print("2. Verifying results actually contain the searched phrase")
    print("3. Testing that unrelated phrases don't match")
    print()
    
    # Initialize search context
    with SearchContext() as ctx:
        print(f"Index loaded: {ctx.num_docs:,} documents")
        
        all_passed = True
        
        # Test 1: Real phrase from document (Kenya rabbits)
        test1 = test_phrase(
            ctx,
            "In Kenya, rabbit keeping is traditionally a hobby for teenage boys",
            expected_in_result=True
        )
        all_passed = all_passed and test1
        
        # Test 2: Real phrase from document (lung cancer)
        test2 = test_phrase(
            ctx,
            "Although the biggest cause of lung cancer is smoking",
            expected_in_result=True
        )
        all_passed = all_passed and test2
        
        # Test 3: Real phrase from document (Rabbit Republic)
        test3 = test_phrase(
            ctx,
            "Every month, Rabbit Republic slaughters about 2,000 rabbits",
            expected_in_result=True
        )
        all_passed = all_passed and test3
        
        # Test 4: Random phrase that should NOT exist (GSM8K question)
        test4 = test_phrase(
            ctx,
            "A robe takes 2 bolts of blue fiber and half that much white fiber",
            expected_in_result=False  # Should NOT be in training data
        )
        all_passed = all_passed and test4
        
        # Test 5: Another GSM8K question that shouldn't exist
        test5 = test_phrase(
            ctx,
            "Natalia sold clips to 48 of her friends in April",
            expected_in_result=False  # Should NOT be in training data
        )
        all_passed = all_passed and test5
        
        # Test 6: MATH question that shouldn't exist
        test6 = test_phrase(
            ctx,
            "Square ABCD has its center at (8,-8) and has an area of 4 square units",
            expected_in_result=False  # Should NOT be in training data
        )
        all_passed = all_passed and test6
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        if all_passed:
            print("✓ ALL TESTS PASSED!")
            print("Phrase search is working correctly.")
            return 0
        else:
            print("❌ SOME TESTS FAILED!")
            print("Phrase search may not be working as expected.")
            return 1


if __name__ == "__main__":
    sys.exit(main())

