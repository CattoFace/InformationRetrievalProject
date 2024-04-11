#!/usr/bin/env python
# coding: utf-8

# Import libraries and get the stop words list

# In[1]:


import pandas as pd
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words("english"))


# Define functions that we will use for all the models

# In[27]:


# Remove all separators using the separators.txt file
def preprocess(text):
    text = text.replace('-', '')
    with open('separators.txt', 'r', encoding='utf8') as f:
        separators = f.read().splitlines()
    for s in separators:
        text = text.replace(s, ' ')
    return text

# create dataframe where columns are docs and rows are words and cells are term frequency in the corresponding document
def create_term_frequency_df(doc_dict):
    term_frequencies = {}
    for filename in doc_dict:
        word_count = {}
        for word in doc_dict[filename]:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        term_frequencies[filename] = word_count
    df = pd.DataFrame(term_frequencies).fillna(0).astype(int)
    return df

# create a language model from the dataframe. We use Laplace smoothing
def language_model(data):
    lengths = data.sum(axis=0)
    for col in data.columns:
        data[col] += 1
        data[col] /= lengths[col] + data.shape[0]
    return data

# for each column we save 20 most popular words(with the hifhest probability) and save them to specified file
def model_to_txt(model, name):
    words_dict = {}
    for col in model.columns:
        file_col = model[col]
        file_col = file_col.sort_values(ascending=False)
        file_col = file_col.head(20)

        popular_words = []
        for word in file_col.index:
            popular_words.append(f'{word}:{round(file_col[word], 7)}')
        words_dict[col] = popular_words

    with open(f'{name}.txt', 'w', encoding='utf8') as f:
        for key, value in words_dict.items():
            f.write(f'{key}:{value}\n')

# functions that gets all the documents as dictionary and creates a text file with the specified name that contains 20 most popular word for 
# each document in the collection
def lm_result(docs, filename):
    term_freq = create_term_frequency_df(docs)
    print(f'Total number of tokens is: {term_freq.sum().sum()}')
    print(f'Vocabulary size of the {filename} is: {term_freq.shape[0]}')
    lm = language_model(term_freq)
    model_to_txt(lm, filename)


# Preprocess the documents and analize the language model result.

# In[28]:


documents = {}
for filename in os.listdir('text'):
  file_path = os.path.join('text', filename)
  with open(file_path, encoding='utf8') as file:
    text = file.read()
    text = preprocess(text)
    text = word_tokenize(text)
    text = [word for word in text if len(word) > 2]
    documents[filename] = text
lm_result(documents, 'collection_model')


# Now remove stop words and make the same

# In[29]:


no_stop_words_docs = documents
for filename in no_stop_words_docs:
    no_stop_words_docs[filename] = [word for word in no_stop_words_docs[filename] if word not in stop_words]
lm_result(no_stop_words_docs, 'no_stop_words_model')


# Perform Case Folding

# In[30]:


case_fold_docs = no_stop_words_docs
for filename in case_fold_docs:
    case_fold_docs[filename] = [word.casefold() for word in case_fold_docs[filename] if word.casefold() not in stop_words]
lm_result(case_fold_docs, 'case_folding_model')


# Use Porter Stemming

# In[31]:


ps = PorterStemmer()
stemmed_docs = case_fold_docs
for filename in stemmed_docs:
    stemmed_docs[filename] = [ps.stem(word) for word in stemmed_docs[filename]]
lm_result(stemmed_docs, 'stemmed_model')

