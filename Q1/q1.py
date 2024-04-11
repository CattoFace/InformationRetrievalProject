import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from nltk.stem import PorterStemmer

np.random.seed(42)

data = pd.read_parquet("data.pk")
stemmer = PorterStemmer()


class StemmingCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmingCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(word) for word in analyzer(doc)])


class StopwordsOnlyCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StopwordsOnlyCountVectorizer, self).build_analyzer()
        return lambda doc: (
            [
                word
                for word in analyzer(doc)
                if word.lower() not in self.get_stop_words()
            ]
        )


def analyze(name: str, vectorizer: CountVectorizer):
    transformed = vectorizer.fit_transform(data["text"])
    voc = vectorizer.get_feature_names_out()
    voc_size = len(voc)
    transformed0 = transformed.sum(0)
    token_count = transformed0.sum()
    top20_indices = transformed0.argsort()[0, ::-1][0, :20]
    top20 = voc[top20_indices].tolist()[0]
    top20_counts = transformed0[0, top20_indices].tolist()[0]
    top20_df = pd.DataFrame({"Term": top20, "Count": top20_counts})
    print(f"{name}: Vocabulary Size={voc_size}, Token Count={token_count}")
    print("Top 20 words:\n", top20_df)


basic = CountVectorizer(lowercase=False)
stopwords = StopwordsOnlyCountVectorizer(lowercase=False, stop_words="english")
casefolded = CountVectorizer(lowercase=True, stop_words="english")
stemmed = StemmingCountVectorizer(lowercase=True, stop_words="english")
breakpoint()
analyze("Basic", basic)
analyze("No stopwords", stopwords)
analyze("With Casefolding", casefolded)
analyze("With Stemming", stemmed)
