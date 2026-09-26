import sys, time
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from candidate_safety import apply_candidate_safety_cap

# Test on 200 France S1 entities
s1 = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=2000, na_filter=False)
s1['norm_country'] = s1['country'].apply(normalize_country)
s1_fr = s1[s1['norm_country'] == 'france'].head(200).copy()
s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)

s2 = pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, nrows=20000, na_filter=False)
s2['norm_country'] = s2['country'].apply(normalize_country)
s2_fr = s2[s2['norm_country'] == 'france'].copy()
s2_fr['source'] = 'S2'

s3 = pd.read_csv('data/test/test_source3.tsv', sep='\t', dtype=str, nrows=20000, na_filter=False)
s3['norm_country'] = s3['country'].apply(normalize_country)
s3_fr = s3[s3['norm_country'] == 'france'].copy()
s3_fr['source'] = 'S3'

targets = pd.concat([s2_fr, s3_fr], ignore_index=True)
targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
targets['norm_address'] = targets['business_address'].apply(normalize_business_address)

print(f"Loaded {len(s1_fr)} S1 records, {len(targets)} targets.")

# Standard ConfigABlockingEngine
t0 = time.time()
engine = ConfigABlockingEngine(targets)
cands_raw = engine.generate_candidates_with_evidence(s1_fr)
t_block = time.time() - t0

t0 = time.time()
capped_cands, _ = apply_candidate_safety_cap(cands_raw, max_candidates_per_s1=400)
t_cap = time.time() - t0

print(f"Standard Engine: {len(cands_raw):,} raw cands in {t_block:.2f}s, {len(capped_cands):,} capped in {t_cap:.2f}s")

# Now let's implement the per-S1 top-400 directly during blocking and compare
t0 = time.time()
raw_total = 0
overflow_count = 0
fast_records = []

s1_eids = s1_fr['entity_id'].values
s1_names = s1_fr['norm_name'].values
s1_addrs = s1_fr['norm_address'].values
s1_countries = s1_fr['norm_country'].values

for s1_id, name, addr, country in zip(s1_eids, s1_names, s1_addrs, s1_countries):
    cand_evidence = {}
    
    # Channel 2: Name token
    for t in name.split():
        if len(t) > 1 and t not in engine.stop_tokens:
            for idx in engine.country_name_token_idx[country].get(t, []):
                if idx not in cand_evidence:
                    cand_evidence[idx] = {'blocked_country': 1, 'blocked_name_token': 1, 'blocked_prefix_3': 0, 'blocked_prefix_4': 0, 'blocked_address_token': 0, 'evidence_score': 70}
                else:
                    if not cand_evidence[idx]['blocked_name_token']:
                        cand_evidence[idx]['blocked_name_token'] = 1
                        cand_evidence[idx]['evidence_score'] += 70
                        
    # Channel 3: Prefix 4
    if len(name) >= 4:
        p4 = name[:4]
        for idx in engine.country_prefix4_idx[country].get(p4, []):
            if idx not in cand_evidence:
                cand_evidence[idx] = {'blocked_country': 1, 'blocked_name_token': 0, 'blocked_prefix_3': 0, 'blocked_prefix_4': 1, 'blocked_address_token': 0, 'evidence_score': 50}
            else:
                if not cand_evidence[idx]['blocked_prefix_4']:
                    cand_evidence[idx]['blocked_prefix_4'] = 1
                    cand_evidence[idx]['evidence_score'] += 50
                    
    # Channel 4: Prefix 3
    if len(name) >= 3:
        p3 = name[:3]
        for idx in engine.country_prefix3_idx[country].get(p3, []):
            if idx not in cand_evidence:
                cand_evidence[idx] = {'blocked_country': 1, 'blocked_name_token': 0, 'blocked_prefix_3': 1, 'blocked_prefix_4': 0, 'blocked_address_token': 0, 'evidence_score': 30}
            else:
                if not cand_evidence[idx]['blocked_prefix_3']:
                    cand_evidence[idx]['blocked_prefix_3'] = 1
                    cand_evidence[idx]['evidence_score'] += 30
                    
    # Channel 5: Address token
    if addr:
        addr_toks = set([t for t in addr.split() if len(t) >= 4 and t not in engine.stop_addrs])
        for t in addr_toks:
            for idx in engine.country_addr_token_idx[country].get(t, []):
                if idx not in cand_evidence:
                    cand_evidence[idx] = {'blocked_country': 1, 'blocked_name_token': 0, 'blocked_prefix_3': 0, 'blocked_prefix_4': 0, 'blocked_address_token': 1, 'evidence_score': 60}
                else:
                    if not cand_evidence[idx]['blocked_address_token']:
                        cand_evidence[idx]['blocked_address_token'] = 1
                        cand_evidence[idx]['evidence_score'] += 60

    n_cands = len(cand_evidence)
    raw_total += n_cands
    if n_cands > 400:
        overflow_count += 1
        
    # Sort and take top 400
    # evidence_score desc, entity_id_cand asc
    items = list(cand_evidence.items())
    # Sort deterministically by: evidence_score desc, entity_id_cand asc
    items.sort(key=lambda x: (-x[1]['evidence_score'], engine.target_eids[x[0]]))
    if len(items) > 400:
        items = items[:400]
        
    for idx, ev in items:
        exact = 1 if name and name == engine.target_names[idx] else 0
        num_keys = ev['blocked_name_token'] + ev['blocked_prefix_3'] + ev['blocked_prefix_4'] + ev['blocked_address_token']
        fast_records.append({
            'entity_id_s1': s1_id,
            'entity_id_cand': engine.target_eids[idx],
            'source': engine.target_sources[idx],
            'blocked_country': 1,
            'blocked_name_token': ev['blocked_name_token'],
            'blocked_prefix_3': ev['blocked_prefix_3'],
            'blocked_prefix_4': ev['blocked_prefix_4'],
            'blocked_address_token': ev['blocked_address_token'],
            'num_blocking_keys': num_keys,
            'evidence_score': float(ev['evidence_score']),
            'blocked_exact_name': exact,
            'blocked_selective_token': ev['blocked_name_token'],
            'blocked_rare_token': 0,
            'blocked_token_pair': 0,
            'blocked_char_ngram': 0,
            'shared_ngram_count': 0
        })

df_fast = pd.DataFrame(fast_records)
t_fast = time.time() - t0

print(f"Fast Engine: {raw_total:,} raw cands, {len(df_fast):,} capped in {t_fast:.2f}s ({t_block+t_cap:.2f}s before, { (t_block+t_cap)/t_fast:.1f}x speedup)")
print(f"Raw cands match exactly: {raw_total == len(cands_raw)}")
print(f"Overflowing count matches: {overflow_count == 200}")

set_standard = set(zip(capped_cands['entity_id_s1'], capped_cands['entity_id_cand']))
set_fast = set(zip(df_fast['entity_id_s1'], df_fast['entity_id_cand']))
print(f"Fast Capped Candidates match Standard 100%: {set_standard == set_fast} (Diff: {len(set_standard ^ set_fast)})")


