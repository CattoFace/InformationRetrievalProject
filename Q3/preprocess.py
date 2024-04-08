from glob import glob
import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download("stopwords")
nltk.download("punkt")
stop_words = stopwords.words("english")

data = []
for file_name in glob("../PoliticalBias/*.txt"):
    with open(file_name, "r", encoding="ISO-8859-1") as file:
        text = file.read()
        data.append({"label": "poli", "text": text})

for file_name in glob("../NewsFairness/*.txt"):
    with open(file_name, "r", encoding="ISO-8859-1") as file:
        text = file.read()
        data.append({"label": "news", "text": text})

for file_name in glob("../SocialBias/*.txt"):
    with open(file_name, "r", encoding="ISO-8859-1") as file:
        text = file.read()
        data.append({"label": "soci", "text": text})

for file_name in glob("../BiasMitigation/*.txt"):
    with open(file_name, "r", encoding="ISO-8859-1") as file:
        text = file.read()
        data.append({"label": "miti", "text": text})
data = pd.DataFrame(data)
data.to_parquet("data.pk")
