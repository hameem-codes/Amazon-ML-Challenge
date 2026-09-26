"""
Verify Step B logic in prototype
"""
import sys, time
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from collections import defaultdict

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine

# Quick 30k target sample
s2 = pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, nrows=30000, na_filter=False)
targets = s2[s2['country'].str.lower() == 'france'].copy().reset_index(drop=True)
targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
targets['norm_address'] = targets['business_address'].apply(normalize_business_address)
targets['norm_country'] = 'france'
targets['source'] = 'S2'

engine = ConfigABlockingEngine(targets)

s1_test = pd.read_csv('data/test/test_source1.tsv', sep='\t', nrows=5000, dtype=str)
fr_s1 = s1_test[s1_test['country'].str.lower() == 'france'].head(25).copy().reset_index(drop=True)
fr_s1['norm_name'] = fr_s1['business_name'].apply(normalize_business_name)
fr_s1['norm_address'] = fr_s1['business_address'].apply(normalize_business_address)
fr_s1['norm_country'] = 'france'

# Precompute target address token sets
target_addr_token_sets = [
    set(t for t in str(addr).split() if len(t) >= 4 and t not in engine.stop_addrs)
    for addr in engine.target_addrs
]

print("Target address token sets built:", len(target_addr_token_sets))
