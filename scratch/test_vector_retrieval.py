import sys, time, gc, psutil, os
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading targets...")
t0 = time.time()
targets = pd.read_pickle('scratch/targets_france.pkl')
print(f"Loaded in {time.time()-t0:.2f}s")

proc = psutil.Process(os.getpid())
print(f"RAM after load: {proc.memory_info().rss / 1024**2:.1f} MB")

t0 = time.time()
vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=5, max_features=30000, sublinear_tf=True)
target_names_clean = targets['norm_name'].fillna('')
X_target = vec.fit_transform(target_names_clean)
build_time = time.time() - t0
print(f"Index built in {build_time:.2f}s, shape: {X_target.shape}, nnz: {X_target.nnz:,}")
print(f"RAM after index: {proc.memory_info().rss / 1024**2:.1f} MB")

# Test queries
s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=50, na_filter=False)
from normalize import normalize_business_name
s1_names = [normalize_business_name(n) for n in s1_df['business_name']]
s1_vecs = vec.transform(s1_names)

t0 = time.time()
for i in range(10):
    # Sparse dot product
    q = s1_vecs[i]
    scores = np.asarray(X_target.dot(q.T).todense()).ravel()
    top800_idx = np.argpartition(scores, -800)[-800:]
query_time_10 = time.time() - t0
print(f"10 queries took: {query_time_10:.3f}s ({query_time_10/10*1000:.1f} ms/query)")
