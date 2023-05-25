import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# load document set
df = pd.read_csv('data/booksummaries.csv',memory_map=True)
# load saved index
index = faiss.read_index('index/booksummaries-trained.index')
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

def get_results(queries):
    results = {}
    for query in queries:
        results[query] = search(query, top_k=5, index=index, model=model)
    return results