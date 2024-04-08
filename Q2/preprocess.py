from glob import glob
import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download("stopwords")
nltk.download("punkt")
stop_words = stopwords.words("english")

data = []
for file_name in glob("../PoliticalBias/*.txt"):
    with open(file_name, "r") as file:
        text = file.read()
        data.append({"label": "poli", "text": text})

for file_name in glob("../NewsFairness/*.txt"):
    with open(file_name, "r") as file:
        text = file.read()
        data.append({"label": "news", "text": text})
data = pd.DataFrame(data)
data.to_parquet("data.pk")
