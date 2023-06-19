import numpy as np
import pandas as pd
import nltk
import json
import re
import csv
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

def load_data():
  data = []
  with open("data/booksummaries.txt", 'r') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
      data.append(row)

  book_index = []
  book_id = []
  book_author = []
  book_name = []
  summary = []
  genre = []
  a = 1
  for i in tqdm(data):
      book_index.append(a)
      a = a+1
      book_id.append(i[0])
      book_name.append(i[2])
      book_author.append(i[3])
      genre.append(i[5])
      summary.append(i[6])
  df = pd.DataFrame({'Index': book_index, 'ID': book_id, 'BookTitle': book_name, 'Author': book_author,
                        'Genre': genre, 'Summary': summary})
  def clean_data(df):
    df.isna().sum()
    df = df.drop(df[df['Genre'] == ''].index)
    df = df.drop(df[df['Summary'] == ''].index)
    genres_cleaned = []
    for i in df['Genre']:
        genres_cleaned.append(list(json.loads(i).values()))
    df['Genre'] = genres_cleaned
    
    def clean_summary(text):
      text = re.sub("\'", "", text)
      text = re.sub("[^a-zA-Z]"," ",text)
      text = ' '.join(text.split())
      text = text.lower()
      return text
    df['clean_summary'] = df['Summary'].apply(lambda x: clean_summary(x))
    return df
  df = clean_data(df)
  return df

# load document set
df = load_data()
# load saved index
index = faiss.read_index('index/booksummaries-trained.index')
index_zero_shot = faiss.read_index('index/booksummaries.index')
# load fine-tuned model
model = SentenceTransformer('model/search-model')

def fetch_book_info(dataframe_idx):
    info = df.iloc[dataframe_idx]
    meta_dict = {}
    meta_dict['Title'] = info['BookTitle']
    meta_dict['Author'] = info['Author']
    meta_dict['Genre'] = info['Genre']
    meta_dict['Summary'] = info['Summary']
    return meta_dict
    
def search(query, top_k, index, model):
    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k)
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results =  [fetch_book_info(idx) for idx in top_k_ids]
    return results

def get_results(queries, trained=True):
    results = {}
    for query in queries:
        if trained:
          results[query] = search(query, top_k=5, index=index, model=model)
        else:
          results[query] = search(query, top_k=5, index=index_zero_shot, model=model)
    return results
