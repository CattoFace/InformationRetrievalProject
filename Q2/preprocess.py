from glob import glob
import pandas as pd

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
