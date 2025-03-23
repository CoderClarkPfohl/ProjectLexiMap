import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from difflib import SequenceMatcher
import re
import time
from pathlib import Path
import sys  # for flushing stdout

# Add caching system
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Debug helpers
def debug_print(message: str, important: bool = False):
    """Print debug message with timestamp and flush immediately."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    prefix = "🔴" if important else "🔹"
    print(f"{prefix} [{timestamp}] {message}", flush=True)

def normalize_keyword(keyword: str) -> str:
    """Normalize a keyword by removing special characters and converting to lowercase."""
    # Remove special characters and extra whitespace
    keyword = re.sub(r'[^\w\s-]', '', keyword.lower())
    keyword = re.sub(r'\s+', ' ', keyword).strip()
    return keyword

def keyword_similarity(kw1: str, kw2: str) -> float:
    """Calculate similarity between two keywords."""
    # Normalize keywords
    kw1 = normalize_keyword(kw1)
    kw2 = normalize_keyword(kw2)
    
    # If exact match after normalization
    if kw1 == kw2:
        return 1.0
    
    # Check if one is contained in the other
    if kw1 in kw2 or kw2 in kw1:
        return 0.9
    
    # Use sequence matcher for fuzzy matching
    return SequenceMatcher(None, kw1, kw2).ratio()

def find_best_match(keyword: str, candidates: List[str], threshold: float = 0.8) -> Tuple[str, float]:
    """Find the best matching keyword from a list of candidates."""
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        score = keyword_similarity(keyword, candidate)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    
    return best_match, best_score

def load_index_keywords() -> Dict[int, List[str]]:
    """Load keywords from the index_by_chapter.txt file."""
    debug_print("Starting to load index keywords...", important=True)
    
    # Check if cached version exists
    cache_file = CACHE_DIR / "index_keywords.json"
    if cache_file.exists():
        debug_print("Loading index keywords from cache...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            debug_print(f"Successfully loaded {len(result)} chapters from cache")
            return {int(k): v for k, v in result.items()}
    
    debug_print("Processing index keywords (first run)...", important=True)
    keywords = defaultdict(list)
    current_chapter = None
    
    try:
        debug_print("Opening index_by_chapter.txt...")
        with open('textbook/index_by_chapter.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        debug_print(f"Read {len(lines)} lines from index file")
        
        line_count = 0
        for line in lines:
            line_count += 1
            if line_count % 500 == 0:
                debug_print(f"Processing line {line_count}/{len(lines)}")
                
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Chapter '):
                try:
                    current_chapter = int(line.split()[1])
                    debug_print(f"Found chapter {current_chapter}")
                except (IndexError, ValueError) as e:
                    debug_print(f"Error parsing chapter number from line: {line}")
                    continue
            else:
                # any non-empty line that's not a chapter header is a keyword
                if current_chapter is not None and line:
                    keyword = normalize_keyword(line)
                    if keyword:  # only add if not empty after normalization
                        keywords[current_chapter].append(keyword)
        
        debug_print("Finished processing index file")
        
        # print some stats
        for chapter, words in keywords.items():
            debug_print(f"Chapter {chapter}: {len(words)} keywords")
            if words:
                debug_print(f"Sample keywords: {words[:5]}")
        
        # Save to cache
        debug_print("Saving keywords to cache...")
        result = {str(k): v for k, v in keywords.items()}  # Convert int keys to strings for JSON
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        debug_print("Keywords cached successfully")
        
        # Convert back to int keys for return
        return {int(k): v for k, v in result.items()}
    except Exception as e:
        debug_print(f"Error in load_index_keywords: {str(e)}", important=True)
        raise

def load_chapter(chapter_num: int) -> str:
    """load a chapter from the textbook folder."""
    debug_print(f"Loading chapter {chapter_num}...")
    
    # Check if cached version exists
    cache_file = CACHE_DIR / f"chapter_{chapter_num}.txt"
    if cache_file.exists():
        try:
            debug_print(f"Loading chapter {chapter_num} from cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                content = f.read()
                debug_print(f"Chapter {chapter_num} loaded from cache ({len(content)} chars)")
                return content
        except Exception as e:
            debug_print(f"Error reading cached chapter {chapter_num}: {e}")
            # Fall back to original loading if cache read fails
    
    try:
        file_path = f'textbook/ch{chapter_num}.txt'
        debug_print(f"Loading chapter from {file_path}...")
        with open(file_path, 'r') as f:
            content = f.read()
            # Save to cache
            debug_print(f"Saving chapter {chapter_num} to cache...")
            with open(cache_file, 'w', encoding='utf-8') as cf:
                cf.write(content)
            debug_print(f"Chapter {chapter_num} loaded ({len(content)} chars)")
            return content
    except Exception as e:
        debug_print(f"Error loading chapter {chapter_num}: {e}", important=True)
        return ""

def extract_keywords(text: str, keybert: KeyBERT, chapter_num: int = None, top_n: int = 50) -> List[Tuple[str, float]]:
    """extract keywords from text using keybert with caching."""
    debug_print(f"Extracting keywords for {'chapter ' + str(chapter_num) if chapter_num else 'text'}", important=True)
    
    # If chapter_num is provided, try to load from cache
    if chapter_num is not None:
        cache_file = CACHE_DIR / f"keywords_ch{chapter_num}.json"
        if cache_file.exists():
            debug_print(f"Loading keywords for chapter {chapter_num} from cache...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    debug_print(f"Successfully loaded {len(cached_data)} keywords from cache")
                    return [(kw, score) for kw, score in cached_data]
            except Exception as e:
                debug_print(f"Error loading cache: {e}")
                # Continue with extraction if cache loading fails
    
    start_time = time.time()
    debug_print(f"Starting keyword extraction with maxsum (top_n={top_n})...")
    try:
        # Extract single words and phrases
        debug_print("Calling keybert.extract_keywords...")
        keywords = keybert.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=top_n,
            nr_candidates=max(100, top_n * 5),  # ensure enough candidates for maxsum
            use_maxsum=True,
            diversity=0.7
        )
        
        elapsed = time.time() - start_time
        debug_print(f"Extracted {len(keywords)} keywords in {elapsed:.2f} seconds", important=True)
        if keywords:
            debug_print(f"Sample keywords: {keywords[:5]}")
        
        # Save to cache if chapter_num is provided
        if chapter_num is not None:
            debug_print(f"Saving keywords for chapter {chapter_num} to cache...")
            cache_file = CACHE_DIR / f"keywords_ch{chapter_num}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                # Convert numpy floats to Python floats for JSON serialization
                keywords_serializable = [(kw, float(score)) for kw, score in keywords]
                json.dump(keywords_serializable, f, indent=2)
            debug_print("Keywords saved to cache successfully")
        
        return keywords
    except Exception as e:
        debug_print(f"Maxsum extraction failed: {e}", important=True)
        debug_print("Trying fallback extraction method...")
        # fallback to simpler method without maxsum
        try:
            keywords = keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=min(30, top_n),  # reduce number of keywords for fallback
            )
            
            elapsed = time.time() - start_time
            debug_print(f"Extracted {len(keywords)} keywords with fallback method in {elapsed:.2f} seconds", important=True)
            if keywords:
                debug_print(f"Sample keywords: {keywords[:5]}")
            
            # Save to cache if chapter_num is provided
            if chapter_num is not None:
                debug_print(f"Saving fallback keywords for chapter {chapter_num} to cache...")
                cache_file = CACHE_DIR / f"keywords_ch{chapter_num}.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    # Convert numpy floats to Python floats for JSON serialization
                    keywords_serializable = [(kw, float(score)) for kw, score in keywords]
                    json.dump(keywords_serializable, f, indent=2)
                debug_print("Fallback keywords saved to cache successfully")
            
            return keywords
        except Exception as e2:
            debug_print(f"Even fallback method failed: {e2}", important=True)
            debug_print("Returning empty keywords list")
            return []

def calculate_metrics(predicted: List[Tuple[str, float]], ground_truth: List[str], 
                     similarity_threshold: float = 0.8) -> Dict[str, float]:
    """calculate precision, recall, and f1 score using fuzzy matching."""
    debug_print(f"Calculating metrics: {len(predicted)} predicted vs {len(ground_truth)} ground truth keywords")
    start_time = time.time()
    
    matches = []
    used_truth = set()
    
    # Normalize predicted keywords
    debug_print("Normalizing predicted keywords...")
    pred_keywords = [(normalize_keyword(kw), score) for kw, score in predicted]
    
    # For each predicted keyword, find the best matching ground truth keyword
    debug_print("Finding best matches...")
    match_count = 0
    total_predictions = len(pred_keywords)
    
    for i, (pred_kw, pred_score) in enumerate(pred_keywords):
        if i % 20 == 0 or i == total_predictions - 1:
            debug_print(f"Matching keyword {i+1}/{total_predictions} ({(i+1)/total_predictions*100:.1f}%)")
            
        best_match, match_score = find_best_match(pred_kw, ground_truth, similarity_threshold)
        if best_match and best_match not in used_truth:
            matches.append((pred_kw, best_match, match_score))
            used_truth.add(best_match)
            match_count += 1
    
    elapsed = time.time() - start_time
    debug_print(f"Matching complete in {elapsed:.2f} seconds, found {match_count} matches")
    
    true_positives = len(matches)
    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives
    
    debug_print(f"\nMatched keywords:")
    for pred, truth, score in matches[:10]:  # Show first 10 matches
        debug_print(f"  {pred} -> {truth} (score: {score:.3f})")
    
    debug_print(f"\nMetrics:")
    debug_print(f"True positives: {true_positives}")
    debug_print(f"False positives: {false_positives}")
    debug_print(f"False negatives: {false_negatives}")
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    debug_print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matches": [(p, t, s) for p, t, s in matches]
    }

def plot_metrics(metrics: Dict[int, Dict[str, float]], output_file: str = "keybert_evaluation.png"):
    """plot evaluation metrics for each chapter."""
    debug_print("Plotting metrics...", important=True)
    chapters = sorted(metrics.keys())
    precisions = [metrics[ch]["precision"] for ch in chapters]
    recalls = [metrics[ch]["recall"] for ch in chapters]
    f1_scores = [metrics[ch]["f1"] for ch in chapters]
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(chapters))
    width = 0.25
    
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1_scores, width, label='F1 Score')
    
    plt.xlabel('Chapter')
    plt.ylabel('Score')
    plt.title('KeyBERT Performance by Chapter')
    plt.xticks(x, chapters)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    debug_print(f"Saving plot to {output_file}...")
    plt.savefig(output_file)
    plt.close()
    debug_print("Plot saved successfully")

def main():
    overall_start_time = time.time()
    debug_print("Starting evaluation process...", important=True)
    
    # load ground truth keywords from index
    try:
        chapter_keywords = load_index_keywords()
        debug_print(f"Loaded keywords for {len(chapter_keywords)} chapters")
    except Exception as e:
        debug_print(f"Failed to load index keywords: {e}", important=True)
        return
    
    # initialize keybert
    debug_print("Initializing keybert model...", important=True)
    try:
        sentence_model = SentenceTransformer('all-mpnet-base-v2')
        keybert = KeyBERT(model=sentence_model)
        debug_print("KeyBERT initialized successfully")
    except Exception as e:
        debug_print(f"Failed to initialize KeyBERT: {e}", important=True)
        return
    
    # evaluate each chapter
    results = {}
    metrics = {}
    
    total_chapters = len(chapter_keywords.keys())
    processed = 0
    
    for chapter_num in sorted(chapter_keywords.keys()):
        chapter_start_time = time.time()
        processed += 1
        debug_print(f"Processing chapter {chapter_num} ({processed}/{total_chapters})...", important=True)
        
        # get chapter text
        try:
            text = load_chapter(chapter_num)
            debug_print(f"Chapter {chapter_num} text length: {len(text)} characters")
            if not text:
                debug_print(f"Skipping chapter {chapter_num} - no text available", important=True)
                continue
        except Exception as e:
            debug_print(f"Error loading chapter {chapter_num}: {e}", important=True)
            continue
            
        # extract keywords with caching
        try:
            debug_print(f"Extracting keywords for chapter {chapter_num}...")
            keywords = extract_keywords(text, keybert, chapter_num=chapter_num)
            debug_print(f"Extracted {len(keywords)} keywords for chapter {chapter_num}")
        except Exception as e:
            debug_print(f"Error extracting keywords for chapter {chapter_num}: {e}", important=True)
            continue
        
        # calculate metrics
        try:
            debug_print(f"Calculating metrics for chapter {chapter_num}...")
            chapter_metrics = calculate_metrics(keywords, chapter_keywords[chapter_num])
            metrics[chapter_num] = chapter_metrics
            debug_print(f"Chapter {chapter_num} metrics: {chapter_metrics['precision']:.3f} precision, {chapter_metrics['recall']:.3f} recall, {chapter_metrics['f1']:.3f} F1")
        except Exception as e:
            debug_print(f"Error calculating metrics for chapter {chapter_num}: {e}", important=True)
            continue
        
        # store results
        results[f"chapter_{chapter_num}"] = {
            "extracted_keywords": [(kw, float(score)) for kw, score in keywords],  # convert numpy float to python float
            "ground_truth": chapter_keywords[chapter_num],
            "metrics": chapter_metrics
        }
        
        chapter_elapsed = time.time() - chapter_start_time
        debug_print(f"Chapter {chapter_num} processed in {chapter_elapsed:.2f} seconds", important=True)
        
        # Save interim results every few chapters
        if processed % 3 == 0 or processed == total_chapters:
            debug_print(f"Saving interim results after processing {processed}/{total_chapters} chapters...")
            try:
                with open(f"keybert_results_interim_{processed}of{total_chapters}.json", "w") as f:
                    json.dump(results, f, indent=2)
                debug_print("Interim results saved successfully")
            except Exception as e:
                debug_print(f"Error saving interim results: {e}", important=True)
    
    # save detailed results
    debug_print("Saving final evaluation results...", important=True)
    try:
        with open("keybert_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        debug_print("Evaluation results saved successfully")
    except Exception as e:
        debug_print(f"Error saving final results: {e}", important=True)
    
    # plot metrics
    try:
        plot_metrics(metrics)
    except Exception as e:
        debug_print(f"Error plotting metrics: {e}", important=True)
    
    # print average metrics
    debug_print("Calculating average metrics...", important=True)
    avg_metrics = defaultdict(float)
    for m in metrics.values():
        for k, v in m.items():
            if isinstance(v, (int, float)):  # only average numeric metrics
                avg_metrics[k] += v
    
    for k in avg_metrics:
        avg_metrics[k] /= len(metrics)
    
    debug_print("\nAverage metrics across all chapters:", important=True)
    debug_print(f"Precision: {avg_metrics['precision']:.3f}")
    debug_print(f"Recall: {avg_metrics['recall']:.3f}")
    debug_print(f"F1 score: {avg_metrics['f1']:.3f}")
    debug_print(f"Avg true positives: {avg_metrics['true_positives']:.1f}")
    debug_print(f"Avg false positives: {avg_metrics['false_positives']:.1f}")
    debug_print(f"Avg false negatives: {avg_metrics['false_negatives']:.1f}")
    
    total_time = time.time() - overall_start_time
    debug_print(f"\nTotal execution time: {total_time:.2f} seconds", important=True)
    debug_print("Evaluation complete!", important=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        debug_print(f"CRITICAL ERROR: {str(e)}", important=True)
        import traceback
        debug_print(traceback.format_exc(), important=True) 
