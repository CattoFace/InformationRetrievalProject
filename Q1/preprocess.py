from glob import glob
import pandas as pd

data = []
for file_name in glob("../PoliticalBias/*.txt"):
    with open(file_name, "r") as file:
        text = file.read()
        data.append({"text": text})

data = pd.DataFrame(data)
data.to_parquet("data.pk")
