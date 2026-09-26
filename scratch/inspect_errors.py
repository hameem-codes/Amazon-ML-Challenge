import json
import os

with open('scratch/audit_phase5_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

lines = []
lines.append("=== 13 FALSE POSITIVES ===")
for fp in d['false_positives']:
    lines.append(f"\nFP #{fp['index']}:")
    lines.append(f"  S1 ID: {fp['s1_id']}, Target ID: {fp['target_id']} ({fp['target_source']})")
    lines.append(f"  S1 Name: {fp['s1_name']}")
    lines.append(f"  Target Name: {fp['target_name']}")
    lines.append(f"  S1 Addr: {fp['s1_address']}")
    lines.append(f"  Target Addr: {fp['target_address']}")
    lines.append(f"  S1 Country: {fp['s1_country']}, Target Country: {fp['target_country']}")
    lines.append(f"  Model Score: {fp['model_score']:.4f}")
    lines.append(f"  Blocking Keys: {fp['blocking_keys']}, Evidence Score: {fp['evidence_score']}")
    sims = fp.get('similarities', {})
    lines.append(f"  Similarities: Name Jaro={sims.get('name_jaro_winkler', 0):.3f}, Name Jac={sims.get('name_token_jaccard', 0):.3f}, Name TFIDF={sims.get('name_tfidf_cosine', 0):.3f}, Addr Jac={sims.get('address_token_jaccard', 0):.3f}, Addr TFIDF={sims.get('address_tfidf_cosine', 0):.3f}")

lines.append("\n=== 33 FALSE NEGATIVES ===")
for fn in d['false_negatives']:
    lines.append(f"\nFN #{fn['index']}:")
    lines.append(f"  S1 ID: {fn['s1_id']}, Target ID: {fn['target_id']} ({fn['target_source']})")
    lines.append(f"  S1 Name: {fn['s1_name']}")
    lines.append(f"  Target Name: {fn['target_name']}")
    lines.append(f"  S1 Addr: {fn['s1_address']}")
    lines.append(f"  Target Addr: {fn['target_address']}")
    lines.append(f"  S1 Country: {fn['s1_country']}, Target Country: {fn['target_country']}")
    lines.append(f"  Generated: {fn['was_generated']}, Survived Cap: {fn['survived_cap']}")
    lines.append(f"  Model Score: {fn['model_score']:.4f}")
    lines.append(f"  Classification: {fn['classification']} ({fn['classification_desc']})")
    lines.append(f"  Blocking Keys: {fn['blocking_keys']}, Evidence Score: {fn['evidence_score']}")
    sims = fn.get('similarities', {})
    lines.append(f"  Similarities: Name Jaro={sims.get('name_jaro_winkler', 0):.3f}, Name Jac={sims.get('name_token_jaccard', 0):.3f}, Name TFIDF={sims.get('name_tfidf_cosine', 0):.3f}, Addr Jac={sims.get('address_token_jaccard', 0):.3f}, Addr TFIDF={sims.get('address_tfidf_cosine', 0):.3f}")

with open('scratch/error_inspection_utf8.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))

print("Wrote UTF-8 inspection file successfully.")
