import gensim
from gensim.summarization.summarizer import summarize as textrank
from summarizer import Summarizer, TransformerSummarizer

import pandas as pd
import re
import math


##### CONFIG ####
BASELINE = "textrank"
IN_PATH = "./20220420_amazon_reviews_test.csv"
OUT_PATH = f"./{BASELINE}_summaries.csv"
MAX_LENGTH = 70
#################


def concat(arr):
    arr = arr.values.flatten().tolist()
    text = "\n".join(arr)
    return text


def count_sentences(arr):
    max_sents = 0
    num_sents = 0
    for s in arr.values.flatten():
        num_sents += len(re.split("[\.\!\?]", s))
        max_sents = max(max_sents, num_sents)
    return num_sents / arr.shape[0] / max_sents


def gpt2_summarizer(corpus, max_length=70):
    #corpus = [corpus]
    gpt2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
    output = gpt2_model(corpus, min_length=15, max_length=max_length)
    summary = ''.join(output)
    return [summary]


def textrank_summarizer(corpus, word_count=70):
    corpus = [corpus]    
    lst_summaries = [gensim.summarization.summarizer.summarize(txt, word_count=word_count) for txt in corpus]    
    return lst_summaries


def summarizer(data, method):
    df = []
    for row in data.values:
        prod_id = row[0]
        text = row[1]
        max_len = MAX_LENGTH if MAX_LENGTH else int(round(row[7], 0))
        if method == "textrank":
            summary = textrank_summarizer(corpus=text, word_count=max_len)
        elif method == "gpt":
            summary = gpt2_summarizer(corpus=text, max_length=max_len)
        num_sents = len(re.split(r"[\.\!\?]", summary[0]))
        num_words = len(re.split(" +|\s+", summary[0]))
        df.append([prod_id, summary[0], num_sents, num_words])
    df = pd.DataFrame(df, columns=["prod_id", "summary", "num_sents", "word_count"])
    return df


def run():
    raw_data = pd.read_csv(IN_PATH)
    
    df = raw_data[["prod_id", "review", "review_len"]].groupby(["prod_id"], as_index=False).agg(concat).copy()
    
    if not MAX_LENGTH:
        df["min_sents"] = raw_data[["prod_id", "num_sents"]].groupby(["prod_id"], as_index=False).min()["num_sents"]
        df["max_sents"] = raw_data[["prod_id", "num_sents"]].groupby(["prod_id"], as_index=False).max()["num_sents"]
        df["avg_sents"] = raw_data[["prod_id", "num_sents"]].groupby(["prod_id"], as_index=False).mean()["num_sents"]
        df["min_len"] = raw_data[["prod_id", "review_len"]].groupby(["prod_id"], as_index=False).min()["review_len"]
        df["max_len"] = raw_data[["prod_id", "review_len"]].groupby(["prod_id"], as_index=False).max()["review_len"]
        df["avg_len"] = raw_data[["prod_id", "review_len"]].groupby(["prod_id"], as_index=False).mean()["review_len"]
        
    summaries = summarizer(df, method=BASELINE)
    
    summaries.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    run()