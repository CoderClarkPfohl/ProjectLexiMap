import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple
import zipfile

# define train and test chapters
TRAIN_CHAPTERS = [1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19]
TEST_CHAPTERS = [6, 10, 11, 12]

def load_chapter(chapter_num: int) -> str:
    """load a chapter from the textbook folder."""
    try:
        with open(f'textbook/ch{chapter_num}.txt', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"error loading chapter {chapter_num}: {e}")
        return ""

def extract_keywords(text: str, keybert: KeyBERT, top_n: int = 10) -> List[Tuple[str, float]]:
    """extract keywords from text using keybert."""
    return keybert.extract_keywords(text, keyphrase_ngram_range=(1, 3), top_n=top_n)

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
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    keybert = KeyBERT(model=sentence_model)
    
    # load training chapters
    train_texts = []
    for chapter in TRAIN_CHAPTERS:
        text = load_chapter(chapter)
        if text:
            train_texts.append(text)
    
    # combine training texts
    combined_train_text = " ".join(train_texts)
    
    # extract keywords from training data
    train_keywords = extract_keywords(combined_train_text, keybert)
    
    # evaluate on test chapters
    test_results = {}
    for chapter in TEST_CHAPTERS:
        test_text = load_chapter(chapter)
        if test_text:
            # extract keywords for test chapter
            test_keywords = extract_keywords(test_text, keybert)
            
            # evaluate against training keywords
            similarity = evaluate_keywords(test_keywords, train_keywords)
            test_results[f"chapter_{chapter}"] = {
                "keywords": test_keywords,
                "similarity_score": similarity
            }
    
    # save results
    with open("keybert_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

if __name__ == "__main__":
    main() 
