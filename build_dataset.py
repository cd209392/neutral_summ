import sys

import pandas as pd
import json

import re
import time
import argparse


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


def process(df, category, test_ratio, valid_ratio, max_num_reviews=1000, min_prod_reviews=10, max_prod_reviews=15):
    train_ratio = 1.0 - test_ratio - valid_ratio
    MAX_TRAIN_REVIEWS = np.ceil(max_num_reviews*train_ratio)
    MAX_TEST_REVIEWS = np.ceil(max_num_reviews*test_ratio)
    MAX_VALID_REVIEWS = np.ceil(max_num_reviews*valid_ratio)
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
        if reviews.shape[0] < min_prod_reviews:
            continue

        ratings_counts = reviews["overall"].value_counts()  # number of occurrences of each rating score
        if (dataset != "train" and ratings_counts.shape[0] < 5) or (dataset == "train" and ratings_counts.shape[0] < 4):
            continue

        min_rating_count = int(ratings_counts.min())

        # Filter out product when the lowest number of occurrences of a rating is less than 1/5 of min_prod_reviews
        if min_rating_count < min_prod_reviews//MAX_RATINGS:
            continue

        min_rating_count = int(min(min_rating_count, max_prod_reviews//MAX_RATINGS))

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


def build_dataset(args, max_size=1500000):
    
    # Load data from file
    file_path = args.source
    print(f"Extracting reviews from {file_path}")
        
    df = dict()
    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            tmp_df[idx] = json.loads(line.strip())
            if idx >= max_size:
                break
        print(f"{cat} memory size: {(float(sys.getsizeof(tmp_df)) / 1000000.0):.2f} MB.")
    
    df = process(df, args.category, args.test_ratio, args.valid_ratio, args.max_num_reviews, args.min_prod_reviews, args.max_prod_reviews)
    
    # Reduce number of columns
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
    if df[df["dataset"] == "train"].shape > 0:
        df[df["dataset"] == "train"].to_csv(f"{args.out_dir}train.csv", index=False)
    if df[df["dataset"] == "valid"].shape > 0:
        df[df["dataset"] == "valid"].to_csv(f"{args.out_dir}valid.csv", index=False)
    if df[df["dataset"] == "test"].shape > 0:
        df[df["dataset"] == "test"].to_csv(f"{args.out_dir}test.csv", index=False)
    
    print("Completed!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to raw JSON data")
    parser.add_argument("--category", type=str, default=None, help="Product category")
    parser.add_argument("--test-ratio", dest="test_ratio", type=float, default=0.2, help="Test dataset ratio")
    parser.add_argument("--valid-ratio", dest="valid_ratio", type=float, default=0.2, help="Validation dataset ratio")
    parser.add_argument("--max-size", dest="max_num_reviews", type=int, default=1500, help="Maximum number of reviews for a given category")
    parser.add_argument("--min-prod-reviews", dest="min_prod_reviews", type=int, default=10, help="Minimum number of reviews per product")
    parser.add_argument("--max-prod-reviews", dest="max_prod_reviews", type=int, default=10, help="Maximum number of reviews per product")
    parser.add_argument("--out-dir", dest="out_dir", type=str, default="./data/", help="Output directory")
    # Parse arguments
    args = parser.parse_args()
    build_dataset(args)