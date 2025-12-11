from rank_bm25 import BM25Okapi  # BM25
from .text_preprocess import clean_text


def build_bm25_corpus(texts):
    """ Build BM25 corpus. Clean the text, tokenize it, return a dataframe"""
    # credit given to this user for idea of a BM25 searcher:
    # https://github.com/ev2900/BM25_Search_Example/blob/main/bm25_example.py
    cleaned = [clean_text(t) for t in texts]
    tokenized_corpus = [doc.split() for doc in cleaned]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def search_bm25(bm25, texts, query, top_n):
    """
    given a query, return top_n messages with scores.
    The final variable will be a list or array? Plan on list: List[tuple[integer, float, strign]]:
    takes in the bm25 api, list, string query from user, number of results to send back
    top_n default is 5 for now
    returns dataframe
    """
    # credit given to this user for idea of a BM25 searcher:
    # https://github.com/ev2900/BM25_Search_Example/blob/main/bm25_example.py

    cleaned_query = clean_text(query)
    tokenized_query = cleaned_query.split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = scores.argsort()[::-1][:top_n]

    results = []
    for idx in ranked_indices:
        results.append((int(idx), float(scores[idx]), texts[idx]))
    return results
