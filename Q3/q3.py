import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import rand_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
kf = KFold(10, shuffle=True)

data = pd.read_parquet("data.pk")


# used in interactive mode to see common words
def common_in_docs(docs):
    vectorizer = classifier.steps[0][1]
    vec = vectorizer.transform(docs["text"])
    common = np.array(vec.todense()).mean(0).argsort()[::-1][:20]
    common = vectorizer.get_feature_names_out()[common]
    print(common)


classifier = Pipeline(
    [
        ("tf", TfidfVectorizer(stop_words="english")),
        ("kmeans", KMeans(4, random_state=42)),
    ]
)
result = classifier.fit_predict(data["text"])
most_common = ["news", "soci", "miti", "soci"]
adj_most_common = ["news", "soci", "miti", "poli"]
predictions = list(map(lambda c: most_common[c], result))
clusters = data.groupby(result)
score = rand_score(result, data["label"])
print("rand score: ", score)
ConfusionMatrixDisplay.from_predictions(data["label"].tolist(), predictions)
plt.title("Confusion Matrix")
plt.show()
