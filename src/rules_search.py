"""Approach 1: rules-based keyword search (BM25). No LLM, no tokens."""
import re
import time

from rank_bm25 import BM25Okapi

from src.gemini_client import EMPTY_USAGE
from src.policies import DECLINED, NO_MATCH, SUPPORTED, load_policies

# Words that appear in almost every question or policy and carry no meaning for matching
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "be", "been", "do", "does", "did", "i", "me", "my", "we",
    "our", "you", "your", "it", "its", "of", "to", "in", "on", "for", "at", "by", "with", "about", "and",
    "or", "can", "could", "may", "might", "must", "should", "would", "will", "what", "whats", "which",
    "who", "when", "where", "why", "how", "many", "much", "there", "this", "that", "any", "if", "have",
    "has", "get", "policy", "policies", "company", "employee", "employees", "rule", "rules", "allowed",
}

# Minimum BM25 score for a match to count. Below this the engine says no policy was found
# instead of returning the least-bad policy.
MIN_SCORE = 1.5


def _stem(word: str) -> str:
    """Very light stemming so 'laptops' matches 'laptop' and 'days' matches 'day'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str):
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class RulesSearcher:
    def __init__(self):
        self.df = load_policies()
        corpus = (self.df["title"] + " " + self.df["category"] + " " + self.df["policy_text"]).tolist()
        self.bm25 = BM25Okapi([tokenize(doc) for doc in corpus])

    def search(self, query: str):
        start = time.time()
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens) if tokens else [0.0] * len(self.df)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = float(scores[best_idx])
        elapsed = round(time.time() - start, 4)

        if best_score < MIN_SCORE:
            return {
                "answer": NO_MATCH,
                "policy": "None",
                "category": "N/A",
                "response_time": elapsed,
                "token_usage": dict(EMPTY_USAGE),
                "grounding": DECLINED,
                "score": best_score,
            }

        match = self.df.iloc[best_idx]
        return {
            "answer": match["policy_text"],
            "policy": match["title"],
            "category": match["category"],
            "response_time": elapsed,
            "token_usage": dict(EMPTY_USAGE),
            # The text is copied verbatim from the database, so it is always "supported",
            # but it can still be the wrong policy (measured in the evaluation).
            "grounding": SUPPORTED,
            "score": best_score,
        }
