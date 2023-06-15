import sys

import pandas as pd
import json

import re
import time


ROOT_DATA_PATH = "./all_amazon_review/"
OUT_TRAIN_PATH = "./07032022_amazon_ns_train.csv"
OUT_VALID_PATH = "./07032022_amazon_ns_valid.csv"
OUT_TEST_PATH = "./07032022_amazon_ns_test.csv"

MAX_RATING = 5.0
MIN_RATING = 1.0


def infer_sentiment(rating):
    MIN_POSITIVE = 0.67
    MAX_NEGATIVE = 0.33
    # Scale rating
    rating = (float(rating) - MIN_RATING) / (MAX_RATING - MIN_RATING)

    sentiment = "neutral"

    if rating <= MAX_NEGATIVE:
        sentiment = "negative"
    elif rating >= MIN_POSITIVE:
        sentiment = "positive"

    return sentiment


def load_categories():
    categories = []
    with open("./categories.txt", "r") as f:
        for line in f:
            categories.append(line.strip())
    return categories


def process(df, category, min_reviews=10, max_reviews=15):
    MAX_TRAIN_REVIEWS = 800
    MAX_TEST_REVIEWS = 200
    MAX_VALID_REVIEWS = 200
    MAX_RATINGS = 5

    MAX_NUM_REVIEWS_PER_CATEGORY = MAX_TRAIN_REVIEWS + MAX_VALID_REVIEWS + MAX_TEST_REVIEWS

    dataset = "train"
    
    df = pd.DataFrame.from_dict(df, orient="index")
    
    # Drop reviews without text
    df = df.dropna(subset=["reviewText"])
    df = df[df["reviewText"].apply(lambda x: re.search("[a-zA-Z]+", str(x)) is not None)]

    # Drop duplicates
    df = df.drop_duplicates(subset=["asin", "reviewerID", "reviewText"])
    
    # Add category column
    df["category"] = re.sub("_", " ", category)
    
    # Add sequence lengths column
    df["review_len"] = df["reviewText"].apply(lambda x: len(str(x).split()))
    
    # Filter out reviews with less than 8 words or more than 200 words
    df = df[(df["review_len"] >= 8) & (df["review_len"] <= 200)]

    # Add sentiment information
    st = time.perf_counter()
    df["polarity"] = df["overall"].apply(infer_sentiment)

    # Add dataset type
    df["dataset"] = None

    final_df = None
    num_reviews = 0
    total_reviews = 0
    products = df["asin"].unique()

    for i, asin in enumerate(products):
        reviews = df[df["asin"] == asin].copy()

        # Add dataset type
        reviews["dataset"] = dataset

        # If train data, ignore neutral reviews
        if dataset == "train":
            reviews = reviews[reviews["polarity"] != "neutral"]

        # Shuffle reviews
        reviews = reviews.sample(frac=1).reset_index(drop=True)

        # Filter out products having less than 10 reviews
        if reviews.shape[0] < min_reviews:
            continue

        ratings_counts = reviews["overall"].value_counts()  # number of occurrences of each rating score
        if (dataset != "train" and ratings_counts.shape[0] < 5) or (dataset == "train" and ratings_counts.shape[0] < 4):
            continue

        min_rating_count = int(ratings_counts.min())

        # Filter out product when the lowest number of occurrences of a rating is less than 1/5 of min_reviews
        if min_rating_count < min_reviews//MAX_RATINGS:
            continue

        min_rating_count = int(min(min_rating_count, max_reviews//MAX_RATINGS))

        rating_1 = reviews[reviews["overall"] == 1.0][:min_rating_count]
        rating_2 = reviews[reviews["overall"] == 2.0][:min_rating_count]
        rating_4 = reviews[reviews["overall"] == 4.0][:min_rating_count]
        rating_5 = reviews[reviews["overall"] == 5.0][:min_rating_count]

        if dataset != "train":
            rating_3 = reviews[reviews["overall"] == 3.0][:min_rating_count]
            reviews = pd.concat([rating_1, rating_2, rating_3, rating_4, rating_5])
        else:
            reviews = pd.concat([rating_1, rating_2, rating_4, rating_5])
            
        if final_df is None:
            final_df = reviews
        else:
            final_df = pd.concat([final_df, reviews])
            
        num_reviews += reviews.shape[0]
        total_reviews += reviews.shape[0]

        if dataset == "train" and num_reviews >= MAX_TRAIN_REVIEWS:
            print(f"updated train data: {num_reviews} reviews")
            dataset = "test"
            num_reviews = 0
        elif dataset == "test" and num_reviews >= MAX_VALID_REVIEWS:
            print(f"updated test data: {num_reviews} reviews")
            dataset = "valid"
            num_reviews = 0
        elif dataset == "valid" and num_reviews >= MAX_TEST_REVIEWS:
            print(f"updated valid data: {num_reviews} reviews")
            break
        elif total_reviews >= MAX_NUM_REVIEWS_PER_CATEGORY:
            break
    
    return final_df


def build_dataset(max_size=1500000):
    # Load list of categories
    print("Loading categories..")
    categories = load_categories()
    
    df = None
    
    for cat in categories:
        st = time.perf_counter()

        cat = "_".join(cat.split(" "))
        
        # Load data from file
        file_path = f"{ROOT_DATA_PATH}{cat}.json"
        print(f"Extracting reviews from {file_path}")
        
        tmp_df = dict()
        with open(file_path, "r") as file:
            for idx, line in enumerate(file):
                tmp_df[idx] = json.loads(line.strip())
                if idx >= max_size:
                    break
            
            print(f"{cat} memory size: {(float(sys.getsizeof(tmp_df)) / 1000000.0):.2f} MB.")
            
            tmp_df = process(tmp_df, cat)
            
            if df is None:
                df = tmp_df.copy()
            else:
                df = pd.concat([df, tmp_df])
            
            print(f"data memory size: {(float(sys.getsizeof(df)) / 1000000.0):.2f} MB.  time: {(time.perf_counter() - st):.2f} seconds")

    # Drop unnecessary columns
    df = df[["dataset", "category", "asin", "overall", "polarity", "reviewTime", "reviewText", "review_len"]]

    # Rename columns
    df = df.rename(columns={
        "overall": "rating",
        "reviewTime": "posted_at",
        "reviewerID": "customer_id",
        "asin": "prod_id",
        "reviewText": "review"
    })

    # Set date attribute to proper type
    df["posted_at"] = df["posted_at"].astype("datetime64[ns]")

    # Sort data
    df = df.sort_values(by=["dataset", "category", "prod_id", "posted_at"])

    # Remove helper columns
    df = df[["dataset", "category", "prod_id", "rating", "polarity", "review", "review_len"]]

    # Add review ids
    df["review_id"] = 0

    for prod_id in df["prod_id"].unique():
        n_samples = df[df["prod_id"] == prod_id].shape[0]
        df.loc[df["prod_id"] == prod_id, "review_id"] = list(range(0, n_samples))

    # Split data into train/val/test datasets
    df[df["dataset"] == "train"].to_csv(OUT_TRAIN_PATH, index=False)
    df[df["dataset"] == "valid"].to_csv(OUT_VALID_PATH, index=False)
    df[df["dataset"] == "test"].to_csv(OUT_TEST_PATH, index=False)
    
    print("Completed!")
    
    
if __name__ == "__main__":
    build_dataset()
