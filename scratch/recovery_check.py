import os
import json
import hashlib
import xgboost as xgb

results = {}

# 1. Check models
model_files = [
    "models/xgboost_entity_resolution_phase5_configA.json",
    "models/phase5_configA_feature_schema.json",
    "models/phase5_configA_threshold.json",
    "models/phase5_configA_metadata.json"
]

print("=== CHECKING LOCKED ARTIFACTS ===")
for mf in model_files:
    exists = os.path.exists(mf)
    size = os.path.getsize(mf) if exists else 0
    print(f"File: {mf} | Exists: {exists} | Size: {size:,} bytes")

# Check schema
schema_path = "models/phase5_configA_feature_schema.json"
if os.path.exists(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    print(f"Schema type: {type(schema)}, Feature count: {len(schema)}")
    # Print first few and last few features
    if isinstance(schema, list):
        print(f"Features (first 3): {schema[:3]}")
        print(f"Features (last 3): {schema[-3:]}")

# Check threshold
thresh_path = "models/phase5_configA_threshold.json"
if os.path.exists(thresh_path):
    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_data = json.load(f)
    print(f"Threshold data: {thresh_data}")

# Check metadata
meta_path = "models/phase5_configA_metadata.json"
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    print(f"Metadata config: {meta_data.get('config_name', meta_data.get('config'))}")
    print(f"Metadata fingerprint: {meta_data.get('fingerprint')}")
    print(f"Metadata full keys: {list(meta_data.keys())}")
    for k, v in meta_data.items():
        if k != "features":
            print(f"  {k}: {v}")

# Load XGBoost model
xgb_path = "models/xgboost_entity_resolution_phase5_configA.json"
if os.path.exists(xgb_path):
    try:
        model = xgb.Booster()
        model.load_model(xgb_path)
        print(f"XGBoost model loaded successfully. Num trees: {model.num_boosted_rounds()}, Num features: {model.num_features()}")
    except Exception as e:
        print(f"Failed to load XGBoost model: {e}")

# Check MD5 / Fingerprint
with open(schema_path, "rb") as f:
    schema_bytes = f.read()
schema_fp = hashlib.md5(schema_bytes).hexdigest()
print(f"Schema file MD5: {schema_fp}")

print("\n=== CHECKING PHASE 6 OUTPUT FILES & TEMP ARTIFACTS ===")
check_files = [
    "candidate_pairs.tsv",
    "matching_results.tsv",
    "output/candidate_pairs.tsv",
    "output/matching_results.tsv",
    "experiments/phase6_test_inference_metadata.json",
    "experiments/candidate_pairs.tsv",
    "experiments/matching_results.tsv"
]

for cf in check_files:
    exists = os.path.exists(cf)
    if exists:
        size = os.path.getsize(cf)
        print(f"FOUND: {cf} (Size: {size:,} bytes)")
    else:
        print(f"NOT FOUND: {cf}")

# Check output/ and experiments/ directory contents
print("\nContents of output/:")
if os.path.exists("output"):
    for root, dirs, files in os.walk("output"):
        for f in files:
            p = os.path.join(root, f)
            print(f"  {p} ({os.path.getsize(p):,} bytes)")
else:
    print("  output directory does not exist")

print("\nContents of experiments/:")
if os.path.exists("experiments"):
    for root, dirs, files in os.walk("experiments"):
        for f in files:
            p = os.path.join(root, f)
            print(f"  {p} ({os.path.getsize(p):,} bytes)")
else:
    print("  experiments directory does not exist")

print("\nContents of scratch/:")
if os.path.exists("scratch"):
    for root, dirs, files in os.walk("scratch"):
        for f in files:
            p = os.path.join(root, f)
            print(f"  {p} ({os.path.getsize(p):,} bytes)")

