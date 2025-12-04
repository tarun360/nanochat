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


# Test documents extracted from the index using extract_test_documents.py
# These are real documents that exist in the index, used for deterministic testing
TEST_DOCUMENTS = [
    "Owl Pencil Sketch – The purpose of sketches is for you to record some important information for the study objective in the long term. It's like a rough work of final item and it is a free hand drawing several overlapping traces. Sketches are generally series associated with many disconnected traces in which produce to develop a picture. The tools used to be able to create sketches are pencil, pen, water-colour, clays and many more. The the majority of more suitable painting tool is usually pencil by most music artists through the day when graphite has been found.\nLet's begin by deciding on some drawing paper for Owl Pencil Sketch. The visit to the art supplies store will notify you that there's a wide range of art paper available. Some are numerous purpose drawing paper in which you can use for pencil, charcoal, watercolor or perhaps chemicals. Those are more likely to be costlier.\nOnce you have your drawing paper, you'll require to equip yourself along with some pencils. I in person prefer to work along with 2B to 8B pencils. These pencils give myself a wide range of ideals and will allow me to create Owl Pencil Sketch just about any effect I want.",
    "A serological investigation of pestiviruses in sheep in eastern border of Turkey\nAkkan, Hasan Altan\nMetadataShow full item record\nAll pestiviruses are important veterinary pathogens causing economic losses in cattle, sheep, and pigs. In this study, blood samples randomly collected from 465 sheep were analysed for the presence of antibodies to pestiviruses (bovine viral diarrhea virus, border disease virus) using an enzyme-linked immunosorbent assay in the province of Van and their towns. The seroprevalance were estimated as 75.9% and 60.0-82.5% in the sampled animals and sampled towns, respectively. The results revealed that pestiviruses are important abort pathogens in the province of Van and their towns.",
    "The chemical formula for water vapor is H2O. Water is a unique compound because it has the ability to exist on the earth's surface in all three forms: liquid, solid (ice), and gas (water vapor).Continue Reading\nWater vapor is the gas phase of water, and is invisible. The measure of the amount of water vapor suspended in the air is what is known as humidity. When water vapor condenses, it forms droplets, which turn into fog and clouds. If the droplets become large enough, they will fall to the earth as rain. At temperatures less than 32 degrees Fahrenheit, these droplets turn into sleet, ice, hail or snow.\nWater vapor accounts for approximately 90 percent of the Earth's greenhouse effect. It keeps the temperature warm enough to support life.Learn more about Chemistry",
    "Is your child overweight?\nThis is a common concern parents will address with me. Here are some things to consider. -Are you overweight? It is rare that a child will be overweight when their parents are at healthy body weights. Likely because they will mirror your eating and movement behaviours -What is their eating environment like? The same healthy eating environment that works for adults works for kids. And as the parent it's up to you to set your child up with good habits early on. Environmental factors that tend to make the biggest difference include: 1) Avoiding DISTRACTIONS while eating (no tablet, phone, tv, games, movies, etc.), just focus on the food and people you're eating with. 2) Eating SLOW. Meals should last 20+ minutes. 3) Stopping at 80% full/satisfied. They shouldn't feel full, uncomfortable, or like they couldn't go do their favorite activity after eating. -Are they active doing things THEY enjoy 60+ minutes every day (honestly from the experience with my own 4 children I think 2 hours is much better as their mood and sleep all seem better). The environmental habits are likely the most important paired with doing activities daily (for long enough) that your child likes. Two other key habits that will likely go a long way though is to include 1 palm sized portion of lean protein at each meal, 1-2 fist sized portion of veggies, and 1 fist sized portions of fruit or whole minimally processed carbohydrate. For the love of God, please don't tell your child they are over weight, need to watch what they eat, exercise more, etc. Simply love them, model good behaviors, set their plate and eating environment up for success, and support the activity that they enjoy. Whats your biggest struggle when it comes to feeding your kids?",
    "Sweating is a strategy used by many animals, including humans, to cool\noff without expending loads of energy. So why not do the same for\nResearchers at the Swiss Federal Institute of Technology in Zurich, led by Aline Rotzetter, have developed a special polymer that soaks up water in the rain and \"sweats\" when it gets warm. The evaporating water works to cool the house, eliminating a lot of the work of an air conditioner and saving energy in the process.\nThe polymer is called Poly(N-isopropylacrylamide, or PNIPAM.\nIt is made into a mat and covered by a membrane that allows water to\nsoak through it. When it rains, the mat acts like a sponge, soaking up\nwater. But put it in direct sunlight at a temperature of 32 degrees\nCentigrade (89.6 degrees Fahrenheit), and it shrinks while taking on\nhydrophobic properties, squeezing the water out, essentially sweating.\nThe mats were tested on small, model houses — the size of those used on model train sets — and was able to cool more efficiently than conventional polymers. It also insulated the houses so that they heated up more slowly.\nThe next step is to test the mats in the cold — it is not clear yet how they might react to being frozen. Even so, if it can be made on a large scale such mats would be useful for people who live in rainy, tropical areas where there is a lot of rain and heat, and where air conditioning is expensive to install.\nThe research was published online in the journal Advanced Materials.\nCredit: Aline Rotzetter / Advanced Materials"
]


def test_ngram_phrase(ctx, phrase, ngram_size=10, expected_in_result=True):
    """Test n-gram phrase search."""
    print(f"\n{'='*80}")
    print(f"Testing n-gram search (size={ngram_size}): '{phrase[:80]}...'")
    print('='*80)
    
    result = ctx.search_phrase_ngrams(phrase, ngram_size=ngram_size, max_results=1, slop=0)
    
    # Verify result structure
    if 'max_score' not in result or 'best_match' not in result or 'ngram_details' not in result:
        print("❌ ERROR: Invalid result structure!")
        print(f"   Expected keys: max_score, best_match, ngram_details")
        print(f"   Got keys: {list(result.keys())}")
        return False
    
    max_score = result['max_score']
    best_match = result['best_match']
    ngram_details = result['ngram_details']
    
    if not ngram_details:
        print("❌ ERROR: ngram_details is empty!")
        return False
    
    ngram_detail = ngram_details[0]
    position = ngram_detail.get('position', 'unknown')
    matched = ngram_detail.get('matched', False)
    
    print(f"  Search mode: {position}")
    print(f"  Max score: {max_score:.1%}")
    print(f"  Matched: {matched}")
    print(f"  Match count: {ngram_detail.get('match_count', 0)}")
    
    if position == 'full':
        print(f"  Note: Phrase too short, fell back to full phrase search")
        if 'ngram_terms' in ngram_detail and ngram_detail['ngram_terms']:
            print(f"  Matched n-gram terms: {ngram_detail['ngram_terms']}")
    elif position == 'sliding_window':
        print(f"  Note: Using sliding window n-gram search")
    
    if max_score == 0.0 or not matched or best_match is None:
        print("❌ NO RESULTS FOUND")
        if expected_in_result:
            print("   ERROR: Expected to find this phrase!")
            return False
        else:
            print("   ✓ Correctly found no matches")
            return True
    
    print(f"\n✓ Found match:")
    print(f"  Score: {best_match['score']:.1%}")
    print(f"  Location: file={best_match['file_idx']}, rg={best_match['rg_idx']}, doc={best_match['doc_idx']}")
    
    # Check if any part of the phrase appears in the result
    # For n-gram search, we don't expect the full phrase, but at least some n-gram should match
    phrase_lower = phrase.lower()
    text_lower = best_match['text'].lower() if best_match['text'] else ""
    
    # Check if at least a significant portion of the phrase appears
    # For n-gram search, we expect at least one n-gram to match
    words = phrase_lower.split()
    if len(words) >= ngram_size:
        # Check if any n-gram from the phrase appears in the text
        found_ngram = False
        for i in range(len(words) - ngram_size + 1):
            ngram_text = ' '.join(words[i:i + ngram_size])
            if ngram_text in text_lower:
                found_ngram = True
                print(f"  Contains n-gram: ✓ YES")
                print(f"    Matched n-gram: '{ngram_text[:100]}...'")
                # Show context
                idx = text_lower.find(ngram_text)
                start = max(0, idx - 50)
                end = min(len(text_lower), idx + len(ngram_text) + 50)
                context = best_match['text'][start:end]
                print(f"    Context: ...{context}...")
                break
        
        if not found_ngram:
            print(f"  Contains n-gram: ❌ NO")
            if expected_in_result:
                print("\n❌ ERROR: No n-gram from phrase found in result!")
                return False
    else:
        # Short phrase - should match fully
        contains_phrase = phrase_lower in text_lower
        print(f"  Contains exact phrase: {'✓ YES' if contains_phrase else '❌ NO'}")
        if contains_phrase:
            idx = text_lower.find(phrase_lower)
            start = max(0, idx - 50)
            end = min(len(text_lower), idx + len(phrase_lower) + 50)
            context = best_match['text'][start:end]
            print(f"  Context: ...{context}...")
        elif expected_in_result:
            print("\n❌ ERROR: Phrase not found in result!")
            return False
    
    if expected_in_result:
        print("\n✓ Test passed!")
        return True
    else:
        print("\n❌ ERROR: Phrase found but should not have been!")
        return False


def main():
    print("="*80)
    print("PHRASE SEARCH VERIFICATION TEST")
    print("="*80)
    print("\nThis script tests that phrase search works correctly by:")
    print("1. Searching for known phrases from indexed documents")
    print("2. Verifying results actually contain the searched phrase")
    print("3. Testing that unrelated phrases don't match")
    print("4. Testing n-gram search functionality")
    print()
    
    # Initialize search context
    with SearchContext() as ctx:
        print(f"Index loaded: {ctx.num_docs:,} documents")
        
        all_passed = True
        
        # =====================================================================
        # Full Phrase Search Tests
        # =====================================================================
        print("\n" + "="*80)
        print("FULL PHRASE SEARCH TESTS")
        print("="*80)
        
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
        
        # =====================================================================
        # N-gram Search Tests
        # =====================================================================
        print("\n" + "="*80)
        print("N-GRAM SEARCH TESTS")
        print("="*80)
        print("\nUsing real documents extracted from the index for testing...")
        
        # Test 7: Short phrase (should fall back to full phrase search)
        # Use a known phrase that exists
        test7 = test_ngram_phrase(
            ctx,
            "In Kenya, rabbit keeping is traditionally a hobby",
            ngram_size=10,
            expected_in_result=True
        )
        all_passed = all_passed and test7
        
        # Test 8: Long phrase from real document (should match with sliding window)
        # Use first 500 chars of first test document
        test8 = test_ngram_phrase(
            ctx,
            TEST_DOCUMENTS[0][:500].strip(),
            ngram_size=10,
            expected_in_result=True
        )
        all_passed = all_passed and test8
        
        # Test 9: Medium phrase from real document
        # Use second test document (full text, it's relatively short)
        test9 = test_ngram_phrase(
            ctx,
            TEST_DOCUMENTS[1].strip(),
            ngram_size=10,
            expected_in_result=True
        )
        all_passed = all_passed and test9
        
        # Test 10: Longer excerpt from real document
        # Use first 800 chars of third test document
        test10 = test_ngram_phrase(
            ctx,
            TEST_DOCUMENTS[2][:800].strip(),
            ngram_size=10,
            expected_in_result=True
        )
        all_passed = all_passed and test10
        
        # Test 11: Test with different n-gram size (13 instead of 10)
        # Use first 600 chars of fourth test document
        test11 = test_ngram_phrase(
            ctx,
            TEST_DOCUMENTS[3][:600].strip(),
            ngram_size=13,
            expected_in_result=True
        )
        all_passed = all_passed and test11
        
        # Test 12: Long phrase that should NOT exist (GSM8K question - extended)
        test12 = test_ngram_phrase(
            ctx,
            "A robe takes 2 bolts of blue fiber and half that much white fiber. If each bolt costs $15 and the robe requires 3 hours of labor at $20 per hour, what is the total cost of making the robe?",
            ngram_size=10,
            expected_in_result=False  # Should NOT be in training data
        )
        all_passed = all_passed and test12
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        if all_passed:
            print("✓ ALL TESTS PASSED!")
            print("Phrase search and n-gram search are working correctly.")
            return 0
        else:
            print("❌ SOME TESTS FAILED!")
            print("Phrase search or n-gram search may not be working as expected.")
            return 1


if __name__ == "__main__":
    sys.exit(main())

