import pandas as pd
import numpy as np
import os
import ast
from normalize import normalize_business_name, normalize_country
from blocking import block_exact_name, block_country_name, block_token, block_ngram

def parse_matched_ids(val):
    if pd.isna(val) or str(val).strip() == '' or str(val).strip() == '[]':
        return []
    val_str = str(val).strip()
    if val_str.startswith('[') and val_str.endswith(']'):
        try:
            return ast.literal_eval(val_str)
        except:
            pass
    if ',' in val_str:
        return val_str.split(',')
    if ' ' in val_str:
        return val_str.split(' ')
    return [val_str]

def load_data():
    print("Loading 500 S1 records...")
    s1 = pd.read_csv('data/train/train_source1.tsv', sep='\t', dtype=str, nrows=500, na_filter=False)
    s1['norm_name'] = s1['business_name'].apply(normalize_business_name)
    s1['norm_country'] = s1['country'].apply(normalize_country)
    
    gt = pd.read_csv('data/train/train_ground_truth.tsv', sep='\t', dtype=str, na_filter=False)
    gt_s1 = gt[gt['source1_entity_id'].isin(s1['entity_id'])]
    
    true_s2_ids = set()
    true_s3_ids = set()
    true_pairs = set()
    
    for _, row in gt_s1.iterrows():
        s1_id = row['source1_entity_id']
        matches = parse_matched_ids(row['matched_entity_ids'])
        for m in matches:
            if m.startswith('S2'):
                true_s2_ids.add(m)
            elif m.startswith('S3'):
                true_s3_ids.add(m)
            true_pairs.add((s1_id, m))
            
    print(f"Total true pairs for these 500 S1 records: {len(true_pairs)}")
    
    print("Loading subset of S2 (10k + true matches)...")
    s2_bg = pd.read_csv('data/train/train_source2.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    # Get true matches not in background
    s2_chunks = pd.read_csv('data/train/train_source2.tsv', sep='\t', dtype=str, na_filter=False, chunksize=50000)
    s2_true_list = []
    for chunk in s2_chunks:
        true_in_chunk = chunk[chunk['entity_id'].isin(true_s2_ids)]
        s2_true_list.append(true_in_chunk)
        if sum(len(x) for x in s2_true_list) >= len(true_s2_ids):
            break
            
    s2 = pd.concat([s2_bg] + s2_true_list).drop_duplicates(subset=['entity_id'])
    s2['source'] = 'S2'
    s2['norm_name'] = s2['business_name'].apply(normalize_business_name)
    s2['norm_country'] = s2['country'].apply(normalize_country)
    print(f"S2 loaded: {len(s2)}")
    
    print("Loading subset of S3 (10k + true matches)...")
    s3_bg = pd.read_csv('data/train/train_source3.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s3_chunks = pd.read_csv('data/train/train_source3.tsv', sep='\t', dtype=str, na_filter=False, chunksize=50000)
    s3_true_list = []
    for chunk in s3_chunks:
        true_in_chunk = chunk[chunk['entity_id'].isin(true_s3_ids)]
        s3_true_list.append(true_in_chunk)
        if sum(len(x) for x in s3_true_list) >= len(true_s3_ids):
            break
            
    s3 = pd.concat([s3_bg] + s3_true_list).drop_duplicates(subset=['entity_id'])
    s3['source'] = 'S3'
    s3['norm_name'] = s3['business_name'].apply(normalize_business_name)
    s3['norm_country'] = s3['country'].apply(normalize_country)
    print(f"S3 loaded: {len(s3)}")
    
    s23 = pd.concat([s2, s3])
    
    return s1, s23, true_pairs

def evaluate_candidates(candidates, s1_ids, true_pairs, name):
    cand_pairs = set(zip(candidates['entity_id_s1'], candidates['entity_id_cand']))
    
    retrieved_true = cand_pairs.intersection(true_pairs)
    recall = len(retrieved_true) / len(true_pairs) if len(true_pairs) > 0 else 0
    
    cand_counts = candidates.groupby('entity_id_s1').size()
    for s1_id in s1_ids:
        if s1_id not in cand_counts:
            cand_counts[s1_id] = 0
            
    avg_cands = cand_counts.mean()
    med_cands = cand_counts.median()
    max_cands = cand_counts.max()
    zero_cand_s1 = (cand_counts == 0).sum()
    
    s1_with_true = set([p[0] for p in retrieved_true])
    
    return {
        'Strategy': name,
        'Candidate Pairs': len(candidates),
        'Avg Candidates/S1': f"{avg_cands:.2f}",
        'Median': int(med_cands),
        'Max': int(max_cands),
        'True-Pair Recall': f"{recall:.4f}",
        'S1 with >=1 Match': len(s1_with_true),
        'Zero-Candidate S1': int(zero_cand_s1)
    }

def run_experiment():
    s1, s23, true_pairs = load_data()
    s1_ids = s1['entity_id'].tolist()
    
    results = []
    
    print("Testing Exact Name...")
    cand_exact = block_exact_name(s1, s23)
    results.append(evaluate_candidates(cand_exact, s1_ids, true_pairs, "Exact Name"))
    
    print("Testing Country + Exact Name...")
    cand_country = block_country_name(s1, s23)
    results.append(evaluate_candidates(cand_country, s1_ids, true_pairs, "Country + Name"))
    
    print("Testing Token Blocking...")
    cand_token = block_token(s1, s23)
    results.append(evaluate_candidates(cand_token, s1_ids, true_pairs, "Token"))
    
    print("Testing Character N-gram (n=3)...")
    cand_ngram = block_ngram(s1, s23, n=3)
    results.append(evaluate_candidates(cand_ngram, s1_ids, true_pairs, "Char 3-gram"))
    
    res_df = pd.DataFrame(results)
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/phase3_blocking_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== PHASE 3A: BLOCKING EXPERIMENTS ===\n\n")
        f.write("Dataset:\n")
        f.write(f"- Source 1 records tested: {len(s1)}\n")
        f.write(f"- Source 2+3 records used (true matches + 10k random subset per source): {len(s23)}\n")
        f.write(f"- Total true entity pairs for these S1 records: {len(true_pairs)}\n\n")
        
        f.write(res_df.to_string(index=False))
        f.write("\n\nInterpretation:\n")
        f.write("Exact Name blocking generates very few candidates (high precision, low candidate volume) but suffers from low recall because any typo, formatting difference, or missing word breaks the match.\n")
        f.write("Adding Country to Exact Name blocking slightly reduces candidate volume further but doesn't improve recall.\n")
        f.write("Token blocking significantly increases recall by matching entities sharing at least one word, but the candidate volume explodes (due to common tokens like 'Inc', 'LLC', 'Private').\n")
        f.write("Character 3-gram blocking achieves the highest recall, finding matches even with typos, but produces an astronomical number of candidates, making it computationally expensive without filtering.\n")
        
    print("\nExperiment completed. Results saved to experiments/phase3_blocking_report.txt")

if __name__ == '__main__':
    run_experiment()
