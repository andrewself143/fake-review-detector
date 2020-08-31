import torch
import torchtext

from torchtext.datasets import text_classification
NGRAMS = 2
import os

import torch.nn as nn
import torch.nn.functional as F

import op_spam
import yelp
import hauyi

import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.data import Field, BucketIterator, TabularDataset

# reviews, labels = op_spam.parse_op_spam()
# reviews, labels = yelp.get_chi_reviews()
reviews, labels = hauyi.read_chinese()
print(len(reviews), len(labels))


raw_data = {
    "text": [r for r in reviews],
    "label": [l for l in labels],
}

df = pd.DataFrame(raw_data, columns=["text", "label"])

# create train and test set
# train, test = train_test_split(df, test_size=0.1)

# train.to_json("train.json", orient="records", lines=True)
# test.to_json("test.json", orient="records", lines=True)

df.to_csv("data/hauyi.csv")
# test.to_csv("data/op_spam.csv")
