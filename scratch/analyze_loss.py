import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import pandas as pd
from collections import defaultdict
from normalize import normalize_business_name, normalize_country, normalize_business_address

print("Loading cached targets...")
targets = pd.read_pickle('scratch/targets_france.pkl')
s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)

s1_name_map = defaultdict(list)
for idx, r in s1_fr.iterrows():
    s1_name_map[r['norm_name']].append(r['entity_id'])

print("Building exact matches...")
exact_matches_set = set()
t_names = targets['norm_name'].values
t_eids = targets['entity_id'].values
for idx in range(len(t_names)):
    t_name = t_names[idx]
    if t_name in s1_name_map:
        t_eid = t_eids[idx]
        for sid in s1_name_map[t_name]:
            exact_matches_set.add((sid, t_eid))

print("Loading candidates...")
df_cands = pd.read_csv('scratch/candidates_e6_a.tsv.gz', sep='\t')
cand_pairs = set(zip(df_cands['entity_id_s1'], df_cands['entity_id_cand']))

retained_true = cand_pairs & exact_matches_set
lost_true = exact_matches_set - cand_pairs
print(f'Total true pairs: {len(exact_matches_set)}')
print(f'Retained true pairs: {len(retained_true)} ({len(retained_true)/len(exact_matches_set):.2%})')
print(f'Lost true pairs: {len(lost_true)}')

lost_by_s1 = defaultdict(int)
for sid, tid in lost_true:
    lost_by_s1[sid] += 1

print(f'Number of S1 entities losing true pairs: {len(lost_by_s1)}')
for sid, cnt in sorted(lost_by_s1.items(), key=lambda x: x[1], reverse=True)[:15]:
    s1_row = s1_fr[s1_fr['entity_id'] == sid].iloc[0]
    total_true_for_sid = sum(1 for s, t in exact_matches_set if s == sid)
    print(f'  {sid} | Name: "{s1_row["norm_name"]}" | Total True: {total_true_for_sid} | Lost: {cnt} | Retained: {total_true_for_sid - cnt}')
