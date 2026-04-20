import requests
import gzip
import json
import random
import pickle
from transformers import DistilBertTokenizerFast

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512


def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0

    response = requests.get(url, stream=True)

    with gzip.open(response.raw, "rt", encoding="utf-8") as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1

            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def prepare_data():

    genre_url_dict = {
        "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
        "children": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
        "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
        "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
        "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
        "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
        "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
        "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
    }

    genre_reviews_dict = {}

    print("Loading dataset...")
    for genre, url in genre_url_dict.items():
        print(f"Loading {genre}")
        genre_reviews_dict[genre] = load_reviews(url)

    train_texts = []
    train_labels = []

    test_texts = []
    test_labels = []

    for genre, reviews in genre_reviews_dict.items():

        reviews = random.sample(reviews, 1000)

        for review in reviews[:800]:
            train_texts.append(review)
            train_labels.append(genre)

        for review in reviews[800:]:
            test_texts.append(review)
            test_labels.append(genre)

    # ✅ Deterministic label order
    label_list = sorted(list(set(train_labels)))

    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}

    train_labels_encoded = [label2id[label] for label in train_labels]
    test_labels_encoded = [label2id[label] for label in test_labels]

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    return (
        train_encodings,
        test_encodings,
        train_labels_encoded,
        test_labels_encoded,
        label2id,
        id2label,
    )
