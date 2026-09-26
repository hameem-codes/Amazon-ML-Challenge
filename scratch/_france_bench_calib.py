"""
Calibration harness for the Phase 6.1 France benchmark (scratch/benchmark_france_scalability.py).

Purpose: measure wall-clock cost of each benchmark stage and the cProfile overhead
on the candidate-generation stage, so the cost of a full 500-S1 cProfile run can be
predicted before committing hours of CPU time.

READ-ONLY with respect to the repository: it imports the existing modules unchanged
(normalize, blocking, config) and replicates the benchmark's exact call sequence.
No source file and no configuration file is modified.

Usage:
    python scratch/_france_bench_calib.py <num_s1> [profile]
        <num_s1> : how many France S1 records to run through candidate generation
        profile  : if given, run the candidate-generation stage inside cProfile
"""

import sys
import os
import time
import gc
import cProfile
import pstats
import io

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

import pandas as pd  # noqa: E402

from normalize import normalize_business_name, normalize_country, normalize_business_address  # noqa: E402
from blocking import ConfigABlockingEngine  # noqa: E402
from config import BLOCKING_CHUNK_SIZE, CANDIDATE_CAP_CONFIG, get_config_fingerprint  # noqa: E402


def log(msg):
    print("[%s] %s" % (time.strftime('%H:%M:%S'), msg), flush=True)


def main():
    num_s1 = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    do_profile = len(sys.argv) > 2 and sys.argv[2] == 'profile'

    log("CALIBRATION: num_s1=%d profile=%s" % (num_s1, do_profile))
    log("Config fingerprint: %s" % get_config_fingerprint())
    log("BLOCKING_CHUNK_SIZE=%s cap=%s" % (BLOCKING_CHUNK_SIZE, CANDIDATE_CAP_CONFIG['max_candidates_per_s1']))

    # ---- Stage 1: target ingestion (identical to benchmark) ----
    t0 = time.time()
    s2_chunks = [
        chunk[chunk['country'].apply(normalize_country) == 'france']
        for chunk in pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)
    ]
    s2_fr = pd.concat(s2_chunks, ignore_index=True)
    s2_fr['source'] = 'S2'
    del s2_chunks
    gc.collect()
    t_s2 = time.time() - t0
    log("S2 read+filter: %d rows in %.2fs" % (len(s2_fr), t_s2))

    t0 = time.time()
    s3_chunks = [
        chunk[chunk['country'].apply(normalize_country) == 'france']
        for chunk in pd.read_csv('data/test/test_source3.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)
    ]
    s3_fr = pd.concat(s3_chunks, ignore_index=True)
    s3_fr['source'] = 'S3'
    del s3_chunks
    gc.collect()
    t_s3 = time.time() - t0
    log("S3 read+filter: %d rows in %.2fs" % (len(s3_fr), t_s3))

    targets = pd.concat([s2_fr, s3_fr], ignore_index=True)
    del s2_fr, s3_fr
    gc.collect()
    log("Total targets: %d" % len(targets))

    # ---- Stage 2: target normalization ----
    t0 = time.time()
    targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
    t_norm_name = time.time() - t0
    t0 = time.time()
    targets['norm_address'] = targets['business_address'].apply(normalize_business_address)
    t_norm_addr = time.time() - t0
    targets['norm_country'] = 'france'
    log("Normalize names: %.2fs | addresses: %.2fs" % (t_norm_name, t_norm_addr))

    # ---- Stage 3: inverted index construction ----
    t0 = time.time()
    engine = ConfigABlockingEngine(targets)
    t_engine = time.time() - t0
    log("Index construction: %.2fs" % t_engine)

    # ---- Stage 4: S1 load (identical to benchmark) ----
    t0 = time.time()
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()
    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    log("S1 load+normalize: %d France S1 in %.2fs" % (len(s1_fr), time.time() - t0))

    # ---- Stage 5: bounded candidate generation (identical call to benchmark) ----
    s1_subset = s1_fr.head(num_s1).copy()
    gc.collect()

    if do_profile:
        profiler = cProfile.Profile()
        t0 = time.time()
        profiler.enable()
        capped, metrics = engine.generate_bounded_candidates(
            s1_subset, max_candidates_per_s1=400, chunk_size=BLOCKING_CHUNK_SIZE, return_metrics=True)
        profiler.disable()
        t_gen = time.time() - t0
        profiler.dump_stats(os.path.join(REPO_ROOT, 'scratch', '_calib_gen_%d.prof' % num_s1))
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats('cumulative').print_stats(12)
        print(s.getvalue(), flush=True)
    else:
        t0 = time.time()
        capped, metrics = engine.generate_bounded_candidates(
            s1_subset, max_candidates_per_s1=400, chunk_size=BLOCKING_CHUNK_SIZE, return_metrics=True)
        t_gen = time.time() - t0

    n_s1 = len(s1_subset)
    log("Candidate generation: %d S1 in %.2fs (%.4fs/S1) | raw=%d post_cap=%d" % (
        n_s1, t_gen, t_gen / max(n_s1, 1), metrics['raw_candidate_count'], metrics['post_cap_candidate_count']))
    log("CALIB_SUMMARY num_s1=%d profile=%s t_s2=%.2f t_s3=%.2f norm_name=%.2f norm_addr=%.2f "
        "index=%.2f s1_load=%.2f gen=%.2f gen_per_s1=%.5f" % (
            n_s1, do_profile, t_s2, t_s3, t_norm_name, t_norm_addr, t_engine,
            time.time(), t_gen, t_gen / max(n_s1, 1)))


if __name__ == '__main__':
    main()
