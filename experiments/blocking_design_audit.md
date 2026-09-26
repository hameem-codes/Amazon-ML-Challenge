# Blocking Design Audit - Errors, Risks & Suggestions

**Scope:** `src/blocking.py`, `src/candidate_safety.py`, `src/config.py`, `src/tournament_engine.py`, `src/training_pairs.py`, and supporting reports (`experiments/final_blocking_tournament_report.md`, `experiments/final_architecture.md`, `experiments/phase4_integration_audit.md`).

**Locked architecture audited:** `CONFIG_A_TEAM_BASELINE`
Channels: `country`, `name_token`, `name_prefix3`, `name_prefix4`, `address_token`
Current config fingerprint: `87f20ceeb84ccc6ea2d48678c7810ac5`

---

## 0. Overall Rating: 7.5 / 10

A genuinely strong, recall-first, evidence-aware design that is let down by wiring/reproducibility defects and a known-but-unfixed efficiency weakness.

| Dimension | Score | Notes |
|---|---|---|
| Recall / effectiveness | 9/10 | 99.96% candidate recall, 34,738 / 34,752 true pairs found (Config A, 10k S1) |
| Correctness & determinism | 8/10 | Deterministic, country-partitioned, dedup-safe; but weight/flag wiring bugs |
| Candidate efficiency | 5/10 | 2,467 cands/S1, ~710 cands per true pair, max 18,450/S1 |
| Code quality / maintainability | 6/10 | Dead/broken legacy paths, documentation drift, dead config keys |
| Evidence & feature integration | 9/10 | Per-channel bitflags preserved end-to-end (verified) |
| Testing / reproducibility | 8/10 | 12/12 relevant tests pass; fingerprint exists; but flag sensitivity is illusory |

---

## 1. Errors & Risks

### ERROR-1 - Configured evidence weights are dead; exact matches can rank below noisy pairs (CRITICAL)

- **Severity:** Critical
- **Location:** `src/blocking.py:110-155` (inline scoring) + `src/candidate_safety.py:19-20` (short-circuit)
- **What happens:**
  - `ConfigABlockingEngine` writes its own `evidence_score` per candidate using inline weights: name-token `70`, prefix-4 `50`, prefix-3 `30`, address-token `60`.
  - This inline score **excludes** the exact-name bonus and the per-key bonus.
  - `compute_evidence_score()` returns the existing column unchanged whenever `evidence_score` is already present:
    ```python
    if (evidence_score in df_candidates.columns):
        return df_candidates[evidence_score].values.astype(np.float32)
    ```
  - Therefore `CANDIDATE_CAP_CONFIG[ranking_weights]` - including `exact_name_weight: 1000` and `num_keys_weight: 100` - **never applies** in the production candidate path.
- **Impact:** The 400/S1 safety cap ranks candidates using a score that does not honour the exact-name priority the configuration declares. A true exact-name match can be ranked below a weaker same-country pair and dropped during overflow (up to 18,450 candidates/S1).
- **Live evidence (reproduced):**
  ```
  entity_id_cand  blocked_exact_name  blocked_name_token  blocked_prefix_3  blocked_prefix_4  blocked_address_token  num_blocking_keys  evidence_score
        S2-EXACT                   1                   1                 1                 1                      0                  3             220   <- true exact match
        S2-NOISE                   0                   1                 1                 1                      1                  4             330   <- different name, shared generic address
  ```
  The exact-name pair (220) ranks **below** the noise pair (330).

### ERROR-2 - Config flags change the fingerprint but not behaviour (HIGH)

- **Severity:** High (reproducibility trap)
- **Location:** `src/config.py:37-49`; `src/blocking.py:58-80` (engine never reads flags)
- **What happens:** `country_blocking`, `name_token_blocking`, `prefix_3_blocking`, `prefix_4_blocking`, `address_token_blocking`, `min_name_token_len`, `min_prefix_len_3`, `min_prefix_len_4`, `min_addr_token_len` are declared but **never read** by `ConfigABlockingEngine`. The engine hardcodes `len(t) > 1`, `name[:3]`, `name[:4]`, `len(t) >= 4`.
- **Impact:** `get_config_fingerprint()` hashes these keys, so toggling a flag *appears* to change the locked configuration while runtime behaviour is identical - a false sense of config control for train/test drift checks.
- **Live evidence (reproduced):**
  ```
  fingerprint before: 87f20ceeb84ccc6ea2d48678c7810ac5
  after setting prefix_3_blocking=False: f2f910722bf13558ebbee4ccdf581fc7   (changed, but engine behaviour is identical)
  ```

### ERROR-3 - Unfiltered prefix channels cause candidate explosion (HIGH)

- **Severity:** High (efficiency)
- **Location:** `src/blocking.py:68-74` (prefix-3 / prefix-4 indices have no document-frequency cap)
- **What happens:** Unlike the (legacy) filtered n-gram channels, prefix keys are not doc-frequency capped, so frequent prefixes generate huge fan-outs.
- **Impact:** Config A: 24,677,222 candidates, avg 2,467/S1, max 18,450/S1, ~710 candidates per recovered true pair. prefix-3 is also largely **subsumed** by prefix-4 (any pair sharing 4 leading chars shares 3), so it adds volume more than unique recall.
- **Acknowledged in:** `experiments/final_blocking_tournament_report.md` ("prefix-3 and prefix-4 generate massive candidate fan-outs for frequent prefixes without doc frequency filtering").

### ERROR-4 - Address-token channel is not name-gated (MEDIUM)

- **Severity:** Medium (precision)
- **Location:** `src/blocking.py:129-135`
- **What happens:** An address token alone (`industrial`, `park`, ...) can create a candidate with a completely different business name, as long as the country matches.
- **Impact:** Precision noise on generic addresses; only mitigated downstream by the cap. Explicitly flagged as a limitation in the tournament report ("Address token matching without name-prefix gating can introduce noise if addresses are overly generic").

### ERROR-5 - Broken legacy paths + documentation drift (MEDIUM)

- **Severity:** Medium
- **Location:** `src/blocking.py:170-330` (`LegacyMultiKeyBlockingEngine`), `src/tournament_engine.py:71,87,134`, `experiments/final_architecture.md`, `experiments/phase4_integration_audit.md`
- **What happens:**
  - When Config A was locked, keys `rare_token_freq_cap`, `ngram_doc_freq_cap`, `min_shared_ngrams`, `token_pair_min_informative`, `ngram_size` were removed from `BLOCKING_CONFIG`, but the legacy engine and `tournament_engine.py` still reference them.
  - `experiments/final_architecture.md` still describes the **old Config B** channels (filtered char 3-gram, selective token, rare token, token-pair, exact name) as production blocking.
  - `experiments/phase4_integration_audit.md` quotes stale fingerprints (`af143f61867ee2fcd7238f76ea4eb09f`, `11eb6aaf2e8e67da2388d1acd564bcf0`) vs. the current `87f20ceeb84ccc6ea2d48678c7810ac5`.
- **Live evidence (reproduced):**
  ```
  LegacyMultiKeyBlockingEngine(df) -> KeyError: rare_token_freq_cap
  ```
- **Impact:** Config B/C/D results in `final_blocking_tournament_report.md` are no longer reproducible from the current tree; misleading architecture docs.

### ERROR-6 - Pure-Python per-pair loops limit scale (LOW)

- **Severity:** Low (performance)
- **Location:** `src/blocking.py:94-165`
- **What happens:** Candidates are scored in nested Python loops (O(S1 x candidates)). ~183s for 10k S1; scales linearly to the full test set.
- **Impact:** Slow end-to-end inference at full scale. `src/tournament_engine.py` already demonstrates a vectorized `np.unique`/`bincount` approach that could be ported.

---

## 2. Suggestions (prioritised)

### S1 - Fix evidence ranking (highest impact; resolves ERROR-1)
Choose ONE source of truth for `evidence_score`:
- **Option A (recommended):** Stop writing `evidence_score` inside `ConfigABlockingEngine`; emit only the evidence bitflags and let `candidate_safety.compute_evidence_score()` own the score using `CANDIDATE_CAP_CONFIG[ranking_weights]` (so `exact_name_weight: 1000` and `num_keys_weight: 100` actually apply).
- **Option B:** Keep the engine score but fold in the missing bonuses, e.g. add `+1000` when `name == c_name` and `+100 * num_keys`.

Also add a regression test asserting that an exact-name candidate outranks a weaker same-country candidate.

### S2 - Wire or remove config flags (resolves ERROR-2)
- Gate each channel on its `*_blocking` flag in `_build_indexes` / `generate_candidates_with_evidence`, and use `min_name_token_len`, `min_prefix_len_3`, `min_prefix_len_4`, `min_addr_token_len` instead of hardcoded literals.
- **OR** delete the unused keys from `BLOCKING_CONFIG` so the fingerprint reflects real behaviour. Do not ship a fingerprint that implies control the code does not have.

### S3 - Tame the prefix explosion (resolves ERROR-3)
- Apply a document-frequency cap to prefix keys, mirroring `ngram_doc_freq_cap` (e.g. drop prefixes present in >5% of the country pool).
- Consider dropping prefix-3 (largely subsumed by prefix-4) or AND-gating prefix hits with a second key.
- Re-run `src/run_tournament.py` to confirm recall holds while candidate volume drops toward Config B-like efficiency (~480 cands / true pair).

### S4 - Gate the address-token channel (resolves ERROR-4)
- Require at least one name-derived key (token or prefix) before an address token can create a candidate, **or** restrict address blocking to rare address tokens.

### S5 - Remove or re-point broken legacy code and sync docs (resolves ERROR-5)
- Delete/repair `LegacyMultiKeyBlockingEngine` and `src/tournament_engine.py`, or restore the config keys they need.
- Update `experiments/final_architecture.md` to describe the Config A channels.
- Refresh the fingerprints in `experiments/phase4_integration_audit.md` to the current `87f20ceeb84ccc6ea2d48678c7810ac5`.

### S6 - Vectorize candidate scoring (resolves ERROR-6)
- Port the `np.unique` / `np.bincount` channel aggregation already present in `src/tournament_engine.py` into the production engine for large-scale inference.

### S7 - Housekeeping
- Consolidate the duplicated generic-token sets (`src/test_blocking_refined.py` re-declares `GENERIC_BUSINESS_TOKENS` instead of importing from `config.py`).
- Keep channel-ablation tooling (`analyze_gt_channels.py`) referenced from the architecture docs.

---

## 3. What the design does well

1. **Recall-first multi-key union** - five country-partitioned channels unioned deterministically reach a **99.96% recall ceiling** (Config A, 10k S1 / 94.5k targets); only 14 of 34,752 true pairs missed.
2. **Evidence preservation** - every pair carries `blocked_name_token`, `blocked_prefix_3`, `blocked_prefix_4`, `blocked_address_token`, `num_blocking_keys`, `evidence_score`, flowing through to the 57-feature model.
3. **Country as an implicit hard partition** - indices are `country -> key -> indices`, so cross-country false merges are structurally impossible (directly tested).
4. **Evidence-ranked safety cap** - the 400/S1 cap cuts volume ~84% (2,467 -> 400) while retaining **98.7%** of true pairs.
5. **Empirical rigor** - a 4-config tournament on identical data/evaluator, with the winner locked into `config.py`, `BLOCKING_CHANNELS`, and a config fingerprint.

---

## 4. Bottom line

The blocking design is conceptually excellent and empirically validated (recall-first + evidence cap is the right strategy for a 99.96% ceiling) and is well-tested on the happy path. It loses points for config that does not control behaviour, an evidence score that ignores the exact-match priority it declares (reproduced), and a known candidate explosion left unmitigated. Fixing **S1-S3** would move this design to roughly **9/10**.

---

## 5. Verification Appendix

### 5.1 Test suite (Config A blocker + architecture)
Run from inside `src` (tests import modules as top-level):

```
cd src
python -m unittest test_phase5_configA test_architecture -v
```

Result: `Ran 12 tests ... OK` (all pass).

### 5.2 Reproduce ERROR-1 (exact match outranked by noise)
```
cd src
python -c "import pandas as pd; from blocking import ConfigABlockingEngine;
tgt=pd.DataFrame([
 {'entity_id':'S2-EXACT','norm_name':'acme alpha','norm_address':'nowhere','norm_country':'india','source':'S2'},
 {'entity_id':'S2-NOISE','norm_name':'acme beta','norm_address':'zzzz industrial park','norm_country':'india','source':'S2'}]);
s1=pd.DataFrame([{'entity_id':'S1-1','norm_name':'acme alpha','norm_address':'zzzz industrial park','norm_country':'india'}]);
e=ConfigABlockingEngine(tgt); c=e.generate_candidates_with_evidence(s1);
print(c[['entity_id_cand','blocked_exact_name','num_blocking_keys','evidence_score']].to_string(index=False))"
```
Expected: `S2-EXACT` = 220, `S2-NOISE` = 330 (exact match ranked lower).

### 5.3 Reproduce ERROR-2 (fingerprint changes, behaviour does not)
```
cd src
python -c "from config import get_config_fingerprint, BLOCKING_CONFIG;
print('before', get_config_fingerprint());
BLOCKING_CONFIG['prefix_3_blocking']=False;
print('after ', get_config_fingerprint())"
```
Expected: `87f20ceeb84ccc6ea2d48678c7810ac5` -> `f2f910722bf13558ebbee4ccdf581fc7`.

### 5.4 Reproduce ERROR-5 (broken legacy engine)
```
cd src
python -c "import pandas as pd; from blocking import LegacyMultiKeyBlockingEngine;
df=pd.DataFrame([{'entity_id':'S2-1','norm_name':'acme ltd','source':'S2'}]);
LegacyMultiKeyBlockingEngine(df)"
```
Expected: `KeyError: rare_token_freq_cap`.

### 5.5 Key metrics referenced
- Current config fingerprint: `87f20ceeb84ccc6ea2d48678c7810ac5`
- Config A recall: 99.96% (34,738 / 34,752 true pairs; 14 missed)
- Config A candidates: 24,677,222 (avg 2,467.70/S1, max 18,450/S1)
- Post-cap (400/S1) recall: 98.66% (retention 98.70%)

---

*Audit generated from a static + dynamic review of the Config A blocking pipeline. All findings above were reproduced against the current working tree.*
