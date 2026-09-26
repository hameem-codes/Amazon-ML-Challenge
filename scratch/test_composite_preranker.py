import sys, os, time, math, heapq
from collections import defaultdict
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import BLOCKING_CHUNK_SIZE, get_config_fingerprint

BIT_COUNTRY = 1 << 0
BIT_NAME = 1 << 1
BIT_P3 = 1 << 2
BIT_P4 = 1 << 3
BIT_ADDR = 1 << 4

def test_preranker_small():
    targets = pd.read_pickle('scratch/targets_france.pkl')
    engine = ConfigABlockingEngine(targets)
    
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=1000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(10).copy().reset_index(drop=True)
    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    
    addr_cap = 50000
    country_addr_tokens = engine.country_addr_token_idx['france']
    high_tokens = {k for k, v in country_addr_tokens.items() if len(v) > addr_cap}
    high_df_postings_sets = {k: set(country_addr_tokens[k]) for k in high_tokens}
    
    log_N = math.log(engine.n_target)
    
    t0 = time.time()
    for idx_s1, r in s1_fr.iterrows():
        name = r['norm_name']
        addr = r['norm_address']
        country = r['norm_country']
        
        s1_tokens = [t for t in name.split() if len(t) > 1 and t not in engine.stop_tokens]
        s1_p3 = name[:3] if len(name) >= 3 else ''
        s1_p4 = name[:4] if len(name) >= 4 else ''
        s1_addr_tokens = set([t for t in addr.split() if len(t) >= 4 and t not in engine.stop_addrs]) if addr else set()
        
        cand_flags = {}
        cand_name_hits = {}
        cand_addr_hits = {}
        cand_rarity = {}
        
        country_name_tokens = engine.country_name_token_idx[country]
        country_p3 = engine.country_prefix3_idx[country]
        country_p4 = engine.country_prefix4_idx[country]
        
        # Channel 2: Name Token
        for t in s1_tokens:
            if t in country_name_tokens:
                postings = country_name_tokens[t]
                df_t = len(postings)
                rarity_t = max(0.0, 1.0 - math.log(max(df_t, 1)) / log_N)
                for idx in postings:
                    cand_flags[idx] = cand_flags.get(idx, 0) | BIT_NAME
                    cand_name_hits[idx] = cand_name_hits.get(idx, 0) + 1
                    if rarity_t > cand_rarity.get(idx, 0.0):
                        cand_rarity[idx] = rarity_t
                        
        # Channel 3: Prefix 4
        if s1_p4 and s1_p4 in country_p4:
            postings = country_p4[s1_p4]
            df_p4 = len(postings)
            rarity_p4 = max(0.0, 1.0 - math.log(max(df_p4, 1)) / log_N)
            for idx in postings:
                cand_flags[idx] = cand_flags.get(idx, 0) | BIT_P4
                if rarity_p4 > cand_rarity.get(idx, 0.0):
                    cand_rarity[idx] = rarity_p4
                    
        # Channel 4: Prefix 3
        if s1_p3 and s1_p3 in country_p3:
            postings = country_p3[s1_p3]
            df_p3 = len(postings)
            rarity_p3 = max(0.0, 1.0 - math.log(max(df_p3, 1)) / log_N)
            for idx in postings:
                cand_flags[idx] = cand_flags.get(idx, 0) | BIT_P3
                if rarity_p3 > cand_rarity.get(idx, 0.0):
                    cand_rarity[idx] = rarity_p3
                    
        # Channel 5: Address Token (low DF)
        low_df_tokens = [t for t in s1_addr_tokens if t in country_addr_tokens and len(country_addr_tokens[t]) <= addr_cap]
        for atok in low_df_tokens:
            postings = country_addr_tokens[atok]
            df_a = len(postings)
            rarity_a = max(0.0, 1.0 - math.log(max(df_a, 1)) / log_N)
            for idx in postings:
                cand_flags[idx] = cand_flags.get(idx, 0) | BIT_ADDR
                cand_addr_hits[idx] = cand_addr_hits.get(idx, 0) + 1
                if rarity_a > cand_rarity.get(idx, 0.0):
                    cand_rarity[idx] = rarity_a
                    
        # High DF address evidence for existing candidates
        if high_df_postings_sets:
            s1_high_tokens = [t for t in s1_addr_tokens if t in high_df_postings_sets]
            for ht in s1_high_tokens:
                postings_set = high_df_postings_sets[ht]
                df_ht = len(country_addr_tokens[ht])
                rarity_ht = max(0.0, 1.0 - math.log(max(df_ht, 1)) / log_N)
                for idx in list(cand_flags.keys()):
                    if idx in postings_set:
                        cand_flags[idx] = cand_flags[idx] | BIT_ADDR
                        cand_addr_hits[idx] = cand_addr_hits.get(idx, 0) + 1
                        if rarity_ht > cand_rarity.get(idx, 0.0):
                            cand_rarity[idx] = rarity_ht
                            
        n_raw = len(cand_flags)
        print(f"S1 {r['entity_id']}: Raw candidates = {n_raw}")
        
        # Now score each candidate with composite pre-ranker
        len_name_s1 = len(name)
        len_addr_s1 = len(addr)
        n_s1_tokens = max(len(s1_tokens), 1)
        n_s1_addrs = max(len(s1_addr_tokens), 1)
        
        target_names = engine.target_names
        target_addrs = engine.target_addrs
        target_lex = engine.target_lex_rank
        
        composite_scores = {}
        for idx, flags in cand_flags.items():
            c_name = str(target_names[idx])
            c_addr = str(target_addrs[idx])
            
            # 1. Name score [0, 1]
            if name and c_name == name:
                name_score = 1.0
            else:
                tok_ratio = min(cand_name_hits.get(idx, 0) / n_s1_tokens, 1.0)
                has_p4 = 1.0 if (flags & BIT_P4) else 0.0
                has_p3 = 1.0 if (flags & BIT_P3) else 0.0
                name_score = 0.70 * tok_ratio + 0.20 * has_p4 + 0.10 * has_p3
                
            # 2. Address score [0, 1]
            if len(s1_addr_tokens) > 0:
                address_score = min(cand_addr_hits.get(idx, 0) / n_s1_addrs, 1.0)
            else:
                address_score = 0.0
                
            # 3. Rarity score [0, 1]
            rarity_score = cand_rarity.get(idx, 0.0)
            
            # 4. Country score [0, 1]
            country_score = 1.0
            
            # 5. Structural score [0, 1]
            len_c_name = len(c_name)
            name_len_ratio = min(len_name_s1, len_c_name) / max(len_name_s1, len_c_name, 1)
            if len_addr_s1 > 0 and len(c_addr) > 0:
                addr_len_ratio = min(len_addr_s1, len(c_addr)) / max(len_addr_s1, len(c_addr), 1)
                structural_score = 0.6 * name_len_ratio + 0.4 * addr_len_ratio
            else:
                structural_score = name_len_ratio
                
            comp_score = (
                0.50 * name_score
                + 0.20 * address_score
                + 0.15 * rarity_score
                + 0.10 * country_score
                + 0.05 * structural_score
            )
            composite_scores[idx] = comp_score
            
        # Top 400 selection: (-composite_score, target_lex_rank)
        K = 400
        if len(cand_flags) <= K:
            selected = list(cand_flags.keys())
        else:
            selected = heapq.nsmallest(K, cand_flags.keys(), key=lambda i: (-composite_scores[i], target_lex[i]))
            
        print(f"  Selected {len(selected)} candidates. Top composite score: {composite_scores[selected[0]]:.4f}, 400th score: {composite_scores[selected[-1]]:.4f}")
        
    print(f"Small test completed in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    test_preranker_small()
