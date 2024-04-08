import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import rand_score
import pandas as pd
import statistics

np.random.seed(42)
kf = KFold(10, shuffle=True)

data = pd.read_parquet("data.pk")


def evaluate_classifier(pipeline_func) -> tuple[list, float, pd.DataFrame]:
    scores = []
    mispredictions = []
    for train_idx, test_idx in kf.split(data):
        breakpoint()
        classifier: Pipeline = pipeline_func()
        train_data = data.iloc[train_idx]
        classifier.fit(train_data["text"], train_data["label"])
        test_data = data.iloc[test_idx]
        predictions = classifier.predict(test_data["text"])
        score = np.mean(predictions == test_data["label"])
        local_mispredictions = test_data.iloc[
            np.where(predictions != test_data["label"])[0]
        ]
        mispredictions.append(local_mispredictions)
        scores.append(score)
    mispredictions = pd.concat(mispredictions)
    return scores, statistics.mean(scores), mispredictions


classifier = Pipeline(
    [
        ("tf", TfidfVectorizer(stop_words="english")),
        ("kmeans", KMeans(4, random_state=42)),
    ]
)
result = classifier.fit_predict(data["text"])
clusters = data.groupby(result)
score = rand_score(result, data["label"])
print("rand score: ", score)
