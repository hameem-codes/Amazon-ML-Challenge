import sys, time
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from normalize import normalize_business_name

targets = pd.read_pickle('scratch/targets_france.pkl')
vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=5, max_features=30000, sublinear_tf=True)
X_target = vec.fit_transform(targets['norm_name'].fillna(''))

s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=500, na_filter=False)
s1_names = [normalize_business_name(n) for n in s1_df['business_name']]
s1_vecs = vec.transform(s1_names)

t0 = time.time()
# Batch dot product for 50 queries
batch_scores = X_target.dot(s1_vecs[:50].T).toarray()
print(f"Batch dot product for 50 queries took: {time.time()-t0:.2f}s, shape: {batch_scores.shape}")
