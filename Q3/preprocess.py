from glob import glob
import pandas as pd

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
