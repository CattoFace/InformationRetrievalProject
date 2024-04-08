import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pandas as pd
import statistics

np.random.seed(42)
kf = KFold(10, shuffle=True)

data = pd.read_parquet("data.pk")


def evaluate_classifier(
    pipeline_func, print_misprediction_proba=False
) -> tuple[list, float, pd.DataFrame]:
    scores = []
    mispredictions = []
    for train_idx, test_idx in kf.split(data):
        classifier: Pipeline = pipeline_func()
        train_data = data.iloc[train_idx]
        classifier.fit(train_data["text"], train_data["label"])
        test_data = data.iloc[test_idx]
        predictions = classifier.predict(test_data["text"])
        score = np.mean(predictions == test_data["label"])
        local_mispredictions = test_data.iloc[
            np.where(predictions != test_data["label"])[0]
        ]
        if print_misprediction_proba and len(local_mispredictions) > 0:
            print(
                "correct:",
                classifier.predict_proba(
                    test_data.iloc[np.where(predictions == test_data["label"])[0]]
                ),
            )
            print(
                "misclassified:", classifier.predict_proba(local_mispredictions["text"])
            )
        mispredictions.append(local_mispredictions)
        scores.append(score)
        breakpoint()
    mispredictions = pd.concat(mispredictions)
    return scores, statistics.mean(scores), mispredictions


def nb_pipeline():
    return Pipeline(
        [("tf", TfidfVectorizer(stop_words="english")), ("multi_nb", MultinomialNB())]
    )


def sgd_pipeline():
    return Pipeline(
        [("tf", TfidfVectorizer(stop_words="english")), ("sgd", SGDClassifier())]
    )


def knn_pipeline():
    return Pipeline(
        [("tf", TfidfVectorizer(stop_words="english")), ("knn", KNeighborsClassifier())]
    )


def forest_pipeline():
    return Pipeline(
        [
            ("tf", TfidfVectorizer(stop_words="english")),
            ("forest", RandomForestClassifier()),
        ]
    )


scores_nb, avg, mispredictions_nb = evaluate_classifier(nb_pipeline, True)
print("nb:\n", scores_nb, f"{avg*100:.2f}%")
mispredictions_nb.to_html("nb_mispredictions.html")
scores_sgd, avg, mispredictions_sgd = evaluate_classifier(sgd_pipeline, False)
print("sgd:\n", scores_sgd, f"{avg*100:.2f}%")
mispredictions_sgd.to_html("sgd_mispredictions.html")
scores_knn, avg, mispredictions_knn = evaluate_classifier(knn_pipeline, True)
print("knn:\n", scores_knn, f"{avg*100:.2f}%")
mispredictions_knn.to_html("knn_mispredictions.html")
scores_forest, avg, mispredictions_forest = evaluate_classifier(forest_pipeline, True)
print("forest:\n", scores_forest, f"{avg*100:.2f}%")
mispredictions_forest.to_html("forest_mispredictions.html")
