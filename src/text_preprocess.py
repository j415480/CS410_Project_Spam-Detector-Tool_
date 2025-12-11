import re
from typing import Iterable, List
from tqdm import tqdm
import time

"""
A lot of this file was guidied by:
https://dylancastillo.co/posts/nlp-snippets-clean-and-tokenize-text-with-python.html

tokenize url, email, emojis and ascii... URLs suddenly made model think it was not spam due to unique url and email
"""
remove_url = re.compile(r"http\S+|www\.\S+")
remove_email = re.compile(r"\S+@\S+")
remove_all_except_az_09_whitespace = re.compile(r"[^a-z0-9\s]+")


def clean_text(text):
    # text by text... ... also implement progress bar as proof of training
    # lowercase, replace urls and emails
    # remove ascii characters, non a-z0-9
    # remove whitespaces

    text = text.lower()
    text = remove_url.sub(" url ", text)
    text = remove_email.sub(" email ", text)
    text = remove_all_except_az_09_whitespace.sub(" ", text)
    # make tidy, remove empty lines \n and extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def batch_clean_text(texts):
    #also implement progress bar as proof of training
    cleaned = []
    print()
    #https://tqdm.github.io/
    for t in tqdm(texts, desc="Cleaning messages, please wait ~5 seconds...", unit="messages", colour="cyan"):
        begin_timer = time.perf_counter()
        cleaned.append(clean_text(t))
        end_timer = time.perf_counter() - begin_timer
        if end_timer < 0.001:
            time.sleep(abs(0.0001 - end_timer))
    print()
    return cleaned
    #return [clean_text(t) for t in texts]
