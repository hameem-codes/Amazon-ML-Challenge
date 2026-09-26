"""
Phase 6.1: France Scalability Candidate-Generation Benchmark
Evaluates memory-safe bounded candidate generation on:
- 500 France S1 records
- ~1,434,993 France target records (S2 + S3)
Tracks runtime, peak RSS memory, raw candidate count, post-cap candidate count,
candidates/S1, median, maximum, zero-candidate S1 count, and overflowing S1 count.
"""

import sys, os, time, gc, json, threading
sys.path.insert(0, 'src')
import psutil
import pandas as pd
import numpy as np

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import BLOCKING_CHUNK_SIZE, CANDIDATE_CAP_CONFIG, get_config_fingerprint


class PeakMemoryTracker:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = self.process.memory_info().rss
        self._running = False
        self._thread = None

    def _track(self):
        while self._running:
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_bytes = self.process.memory_info().rss
        self._running = True
        self._thread = threading.Thread(target=self._track, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        # Final sample
        rss = self.process.memory_info().rss
        if rss > self.peak_bytes:
            self.peak_bytes = rss
        return self.peak_bytes


def run_france_benchmark():
    print("=" * 60)
    print("PHASE 6.1: FRANCE CANDIDATE-GENERATION SCALABILITY BENCHMARK")
    print("=" * 60)
    
    # 0. Verify Config & Fingerprint
    fp = get_config_fingerprint()
    print(f"Config Fingerprint: {fp}")
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"
    print(f"Execution BLOCKING_CHUNK_SIZE: {BLOCKING_CHUNK_SIZE}")
    print(f"Candidate Safety Cap: {CANDIDATE_CAP_CONFIG['max_candidates_per_s1']}")

    # 1. Load France Targets
    print("\n[1/3] Loading France targets from Source 2 and Source 3...")
    t0_targets = time.time()
    
    s2_chunks = [
        chunk[chunk['country'].apply(normalize_country) == 'france']
        for chunk in pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)
    ]
    s2_fr = pd.concat(s2_chunks, ignore_index=True)
    s2_fr['source'] = 'S2'
    del s2_chunks
    gc.collect()
    print(f"  Source 2 France targets: {len(s2_fr):,}")

    s3_chunks = [
        chunk[chunk['country'].apply(normalize_country) == 'france']
        for chunk in pd.read_csv('data/test/test_source3.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)
    ]
    s3_fr = pd.concat(s3_chunks, ignore_index=True)
    s3_fr['source'] = 'S3'
    del s3_chunks
    gc.collect()
    print(f"  Source 3 France targets: {len(s3_fr):,}")

    targets = pd.concat([s2_fr, s3_fr], ignore_index=True)
    del s2_fr, s3_fr
    gc.collect()

    n_targets = len(targets)
    print(f"  Total France targets combined: {n_targets:,} loaded in {time.time()-t0_targets:.2f}s")

    print("  Normalizing France targets...")
    t0_norm = time.time()
    targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
    targets['norm_address'] = targets['business_address'].apply(normalize_business_address)
    targets['norm_country'] = 'france'
    print(f"  Normalized {n_targets:,} targets in {time.time()-t0_norm:.2f}s")

    print("  Building inverted index engine...")
    t0_eng = time.time()
    engine = ConfigABlockingEngine(targets)
    t_eng = time.time() - t0_eng
    
    proc = psutil.Process(os.getpid())
    engine_rss_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"  Inverted index built in {t_eng:.2f}s | Current RSS: {engine_rss_mb:.1f} MB")

    # 2. Load 500 France S1 Records
    print("\n[2/3] Loading 500 France S1 records...")
    t0_s1 = time.time()
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()

    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    
    n_s1 = len(s1_fr)
    print(f"  Loaded {n_s1} France S1 records in {time.time()-t0_s1:.2f}s")
    assert n_s1 == 500, f"Expected 500 France S1 records, got {n_s1}"

    # 3. Run Bounded Candidate Generation Benchmark
    print(f"\n[3/3] Running bounded candidate generation (chunk_size={BLOCKING_CHUNK_SIZE}, cap=400)...")
    
    gc.collect()
    mem_tracker = PeakMemoryTracker(interval=0.05)
    mem_tracker.start()
    
    t0_bench = time.time()
    capped_cands, metrics = engine.generate_bounded_candidates(
        s1_fr,
        max_candidates_per_s1=400,
        chunk_size=BLOCKING_CHUNK_SIZE,
        return_metrics=True
    )
    bench_duration = time.time() - t0_bench
    peak_rss_bytes = mem_tracker.stop()
    peak_rss_mb = peak_rss_bytes / (1024 * 1024)

    # 4. Compile Benchmark Metrics
    raw_count = metrics['raw_candidate_count']
    post_cap_count = metrics['post_cap_candidate_count']
    cands_per_s1 = metrics['candidates_per_s1']
    median_cands = metrics['median_candidates_per_s1']
    max_cands = metrics['max_candidates_per_s1']
    zero_cand_s1 = metrics['zero_candidate_s1_count']
    overflowing_s1 = metrics['overflowing_s1_count']

    print("\n" + "=" * 60)
    print("FRANCE SCALABILITY BENCHMARK RESULTS")
    print("=" * 60)
    print(f"S1 count:                  {n_s1:,}")
    print(f"Target count:              {n_targets:,}")
    print(f"Runtime:                   {bench_duration:.2f} seconds ({n_s1 / bench_duration:.1f} S1/s)")
    print(f"Peak RSS memory:           {peak_rss_mb:.1f} MB")
    print(f"Raw candidate count:       {raw_count:,}")
    print(f"Post-cap candidate count:  {post_cap_count:,}")
    print(f"Candidates/S1:             {cands_per_s1:.2f}")
    print(f"Median candidates/S1:      {median_cands:.1f}")
    print(f"Maximum candidates/S1:     {max_cands}")
    print(f"Zero-candidate S1 count:   {zero_cand_s1}")
    print(f"Overflowing S1 count:      {overflowing_s1}")
    print("=" * 60)

    # Sanity checks
    assert len(capped_cands) == post_cap_count
    assert max_cands <= 400, f"Candidate cap violated: {max_cands} > 400"
    assert post_cap_count > 0, "No post-cap candidates generated"
    assert raw_count >= post_cap_count, "Raw candidate count less than post-cap"
    
    # Check no duplicates
    pairs = set(zip(capped_cands['entity_id_s1'], capped_cands['entity_id_cand']))
    assert len(pairs) == len(capped_cands), "Duplicate candidate pairs found in output"
    
    # Save results json
    results_path = 'experiments/france_scalability_benchmark_results.json'
    results_data = {
        's1_count': n_s1,
        'target_count': n_targets,
        'runtime_seconds': round(bench_duration, 2),
        'throughput_s1_per_sec': round(n_s1 / bench_duration, 2),
        'peak_rss_mb': round(peak_rss_mb, 1),
        'engine_rss_mb': round(engine_rss_mb, 1),
        'raw_candidate_count': raw_count,
        'post_cap_candidate_count': post_cap_count,
        'candidates_per_s1': round(cands_per_s1, 2),
        'median_candidates_per_s1': round(median_cands, 1),
        'maximum_candidates_per_s1': max_cands,
        'zero_candidate_s1_count': zero_cand_s1,
        'overflowing_s1_count': overflowing_s1,
        'chunk_size': BLOCKING_CHUNK_SIZE,
        'candidate_cap': 400,
        'config_fingerprint': fp,
        'status': 'SUCCESS'
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {results_path}")
    return results_data


if __name__ == '__main__':
    run_france_benchmark()
