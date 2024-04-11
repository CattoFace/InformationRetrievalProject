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
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support
import matplotlib.pyplot as plt

np.random.seed(42)
kf = KFold(10, shuffle=True)

data = pd.read_parquet("data.pk")


def evaluate_classifier(name, pipeline_func, print_misprediction_proba=False):
    scores = []
    all_predictions = []
    real_labels = []
    for train_idx, test_idx in kf.split(data):
        classifier: Pipeline = pipeline_func()
        train_data = data.iloc[train_idx]
        classifier.fit(train_data["text"], train_data["label"])
        test_data = data.iloc[test_idx]
        predictions = classifier.predict(test_data["text"])
        all_predictions += predictions.tolist()
        real_labels += test_data["label"].tolist()
        score = np.mean(predictions == test_data["label"])
        local_mispredictions = test_data.iloc[
            np.where(predictions != test_data["label"])[0]
        ]
        if print_misprediction_proba and len(local_mispredictions) > 0:
            print(
                "correct:",
                classifier.predict_proba(
                    test_data.iloc[np.where(predictions == test_data["label"])[0]][
                        "text"
                    ]
                ),
            )
            print(
                "misclassified:", classifier.predict_proba(local_mispredictions["text"])
            )
        scores.append(score)
    plt.title(name)
    print(f"{name}:\n", scores, f"{statistics.mean(scores)*100:.2f}%")
    precision, recall, fscore, _ = precision_recall_fscore_support(
        real_labels, all_predictions, labels=["poli", "news"]
    )
    print(f"{precision=}, {recall=}, {fscore=}")
    ConfusionMatrixDisplay.from_predictions(
        real_labels, all_predictions, labels=["poli", "news"]
    )
    plt.show()


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


evaluate_classifier("NB", nb_pipeline, True)
evaluate_classifier("SGD", sgd_pipeline, False)
evaluate_classifier("KNN", knn_pipeline, True)
evaluate_classifier("Forest", forest_pipeline, True)
