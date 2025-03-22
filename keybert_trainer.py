import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple
import time
from pathlib import Path

# Add caching system (same as evaluator)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# define train and test chapters
TRAIN_CHAPTERS = [1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19]
TEST_CHAPTERS = [6, 10, 11, 12]

def load_chapter(chapter_num: int) -> str:
    """load a chapter from the textbook folder."""
    # Check if cached version exists
    cache_file = CACHE_DIR / f"chapter_{chapter_num}.txt"
    if cache_file.exists():
        try:
            print(f"Loading chapter {chapter_num} from cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading cached chapter {chapter_num}: {e}")
            # Fall back to original loading if cache read fails
    
    try:
        print(f"Loading chapter {chapter_num} from file...")
        with open(f'textbook/ch{chapter_num}.txt', 'r') as f:
            content = f.read()
            # Save to cache
            print(f"Saving chapter {chapter_num} to cache...")
            with open(cache_file, 'w', encoding='utf-8') as cf:
                cf.write(content)
            return content
    except Exception as e:
        print(f"error loading chapter {chapter_num}: {e}")
        return ""

def extract_keywords(text: str, keybert: KeyBERT, chapter_num: int = None, top_n: int = 10) -> List[Tuple[str, float]]:
    """extract keywords from text using keybert with caching support."""
    # If chapter_num is provided, try to load from cache
    if chapter_num is not None:
        cache_file = CACHE_DIR / f"keywords_ch{chapter_num}.json"
        if cache_file.exists():
            print(f"Loading keywords for chapter {chapter_num} from cache...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if len(cached_data) >= top_n:  # Make sure we have enough keywords
                        print(f"Found {len(cached_data)} keywords in cache")
                        return [(kw, score) for kw, score in cached_data[:top_n]]
                    else:
                        print(f"Cache has only {len(cached_data)} keywords, but we need {top_n}")
            except Exception as e:
                print(f"Error reading cache: {e}")
                # Continue with extraction if cache read fails
    
    start_time = time.time()
    print(f"Extracting keywords for {'chapter ' + str(chapter_num) if chapter_num else 'text'}")
    
    try:
        # first try without maxsum
        keywords = keybert.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            top_n=top_n,
            nr_candidates=max(100, top_n * 5)  # ensure enough candidates 
        )
        
        elapsed = time.time() - start_time
        print(f"Extracted {len(keywords)} keywords in {elapsed:.2f} seconds")
        
        # Save to cache if chapter_num is provided
        if chapter_num is not None:
            print(f"Saving keywords for chapter {chapter_num} to cache...")
            cache_file = CACHE_DIR / f"keywords_ch{chapter_num}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                # Convert numpy floats to Python floats for JSON serialization
                keywords_serializable = [(kw, float(score)) for kw, score in keywords]
                json.dump(keywords_serializable, f, indent=2)
            print("Keywords saved to cache")
        
        return keywords
    except Exception as e:
        print(f"first attempt failed: {e}")
        try:
            # second attempt with simpler parameters
            keywords = keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                top_n=min(top_n, 8),
                nr_candidates=max(50, top_n * 4)
            )
            
            # Save to cache if chapter_num is provided (even for fallback method)
            if chapter_num is not None:
                print(f"Saving fallback keywords for chapter {chapter_num} to cache...")
                cache_file = CACHE_DIR / f"keywords_ch{chapter_num}.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    # Convert numpy floats to Python floats for JSON serialization
                    keywords_serializable = [(kw, float(score)) for kw, score in keywords]
                    json.dump(keywords_serializable, f, indent=2)
            
            return keywords
        except Exception as e:
            print(f"second attempt failed: {e}")
            # final fallback with minimal parameters
            keywords = keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 1),
                top_n=min(top_n, 5),
                nr_candidates=max(20, top_n * 3)
            )
            
            # Still try to cache the results
            if chapter_num is not None:
                print(f"Saving minimal keywords for chapter {chapter_num} to cache...")
                cache_file = CACHE_DIR / f"keywords_ch{chapter_num}.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    keywords_serializable = [(kw, float(score)) for kw, score in keywords]
                    json.dump(keywords_serializable, f, indent=2)
            
            return keywords

def evaluate_keywords(predicted: List[Tuple[str, float]], actual: List[Tuple[str, float]]) -> float:
    """evaluate predicted keywords against actual keywords using cosine similarity."""
    # convert to sets of keywords (ignoring scores)
    pred_set = set(kw for kw, _ in predicted)
    actual_set = set(kw for kw, _ in actual)
    
    # calculate jaccard similarity
    intersection = len(pred_set.intersection(actual_set))
    union = len(pred_set.union(actual_set))
    return intersection / union if union > 0 else 0.0

def main():
    # initialize keybert with sentence transformer
    print("initializing keybert...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    keybert = KeyBERT(model=sentence_model)
    
    # Create a special cache entry for the combined training data
    combined_cache_file = CACHE_DIR / "combined_training_keywords.json"
    train_keywords = []
    
    # Check if we have cached combined keywords
    if combined_cache_file.exists():
        print("Loading combined training keywords from cache...")
        try:
            with open(combined_cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if len(cached_data) >= 30:  # we want 30 keywords
                    train_keywords = [(kw, score) for kw, score in cached_data]
                    print(f"Loaded {len(train_keywords)} combined training keywords from cache")
        except Exception as e:
            print(f"Error loading combined cache: {e}")
            # Will proceed to generating the keywords below
    
    if not train_keywords:
        # load training chapters
        train_texts = []
        for chapter in TRAIN_CHAPTERS:
            text = load_chapter(chapter)
            if text:
                print(f"processing chapter {chapter}...")
                train_texts.append(text)
        
        # combine training texts
        combined_train_text = " ".join(train_texts)
        print(f"combined text length: {len(combined_train_text)} characters")
        
        # extract keywords from training data
        print("extracting keywords from combined training data...")
        train_keywords = extract_keywords(combined_train_text, keybert, top_n=30)
        print(f"found {len(train_keywords)} training keywords")
        
        # Save combined keywords to cache
        print("Saving combined training keywords to cache...")
        with open(combined_cache_file, 'w', encoding='utf-8') as f:
            keywords_serializable = [(kw, float(score)) for kw, score in train_keywords]
            json.dump(keywords_serializable, f, indent=2)
    
    print("sample training keywords:", [kw for kw, _ in train_keywords[:5]])
    
    # evaluate on test chapters
    test_results = {}
    for chapter in TEST_CHAPTERS:
        print(f"\nevaluating chapter {chapter}...")
        test_text = load_chapter(chapter)
        if test_text:
            # extract keywords for test chapter (with caching)
            test_keywords = extract_keywords(test_text, keybert, chapter_num=chapter, top_n=30)
            print(f"found {len(test_keywords)} keywords for chapter {chapter}")
            print("sample keywords:", [kw for kw, _ in test_keywords[:5]])
            
            # evaluate against training keywords
            similarity = evaluate_keywords(test_keywords, train_keywords)
            test_results[f"chapter_{chapter}"] = {
                "keywords": [(kw, float(score)) for kw, score in test_keywords],
                "similarity_score": similarity
            }
            print(f"similarity score: {similarity:.3f}")
    
    # save results
    with open("keybert_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

if __name__ == "__main__":
    main() 