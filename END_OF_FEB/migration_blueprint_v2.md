# MIGRATION BLUEPRINT v2 — Chest CT Entity Evaluation System
## Complete Project Context for Session Continuity
### Generated at session end — supersedes all prior blueprints

---

# TABLE OF CONTENTS

1. Project Overview & Current State
2. Schema Design (definitive)
3. File-by-File Architecture
4. Complete Evaluation Pipeline — End-to-End Flow
5. Embedding & Similarity Infrastructure
6. Observation Comparison: The Full 4-Step Pipeline
7. Location Comparison
8. Degree, Presence & Coverage
9. Hungarian Matching (entity-level)
10. LLM Evaluator
11. Rule-Based Validator
12. Composite Score Construction
13. synonym_map.json — Structure, Logic, Current State
14. All Applied Code Changes (chronological)
15. Outstanding Bugs & Fixes Pending
16. What Was Tried and Rejected
17. GT Dataset Benchmarks (GT0–GT6)
18. Known Systematic Biases
19. Session Transfer Checklist

---

# 1. PROJECT OVERVIEW & CURRENT STATE

## 1.1 What This System Does

A **Named Entity Recognition (NER) evaluation pipeline** for chest CT radiology reports. Given the same radiology report, a GT (ground truth) annotation and a model-predicted annotation both produce a list of structured medical entities. This system scores how well the predicted entities match the ground truth.

The challenge is that medical language is highly paraphrastic:
- "pleural effusion" = "pleural fluid" = "fluid accumulation" (at pericardial)
- "left lower lobe atelectasis" = "collapse @ [left lower lobe]" = "volume loss @ [left base]"
- "coronary artery calcification @ [LAD]" = "calcification @ [coronary artery, lad]" (compound split)

A naive exact-match or even simple cosine similarity evaluator fails on these cases.

## 1.2 Current Version: v3.0

**Status:** Working and correctly ranking GT0–GT6 samples with known remaining issues.

**Architecture:**
```
SapBERT Hungarian Matching → TP/FP/FN (F1)
         ↓ (TP pairs only)
Mixed-Method Field Scoring → FL weighted mean
         × Coverage penalty → FL adjusted
         × LLM clinical assessment → Composite Score
         + Rule Violations (separate track)
```

**Key files:**
- `ulti_comp_mainn.py` — main orchestrator, field scoring, synonym/compound logic
- `entity_level_evaluator.py` — SapBERT matching, Hungarian algorithm, contradiction detection
- `llm_evaluator.py` — Gemini Pro clinical scoring
- `rule_based_validator.py` — schema consistency checks
- `synonym_map.json` — 132-entry synonym/compound resolution map
- `workflows/chest_ct_workflow.py` — thoracic ontology normalization

## 1.3 Schema Version Being Used

**NEW 4-field schema** (NOT the old 6-field schema). Every GT and sample file uses this. Do NOT use old-schema logic.

---

# 2. SCHEMA DESIGN (DEFINITIVE)

## 2.1 JSON Structure

```json
{
  "dataset": "mimic-chest-ct",
  "doc_key": 100,
  "report": "FINDINGS: ... full radiology report text ...",
  "entities": [
    {
      "observation": "coronary artery calcification",
      "observation_presence": "present",
      "location": ["left anterior descending artery"],
      "degree": []
    },
    {
      "observation": "effusion",
      "observation_presence": "present",
      "location": ["pleural", "bilateral"],
      "degree": ["mild"]
    },
    {
      "observation": "pneumothorax",
      "observation_presence": "absent",
      "location": [],
      "degree": []
    }
  ]
}
```

## 2.2 Field Definitions

| Field | Type | Valid Values | Notes |
|-------|------|-------------|-------|
| `observation` | string | Free text | The finding name. Merges old `general_finding` + `specific_finding` into ONE field. Examples: "effusion", "atelectasis", "coronary artery calcification", "nodule", "lymphadenopathy" |
| `observation_presence` | string | `"present"` / `"absent"` / `"uncertain"` | Exactly 3 values. Case-sensitive. "absent" = explicitly negated. "uncertain" = equivocal imaging. |
| `location` | array of strings | Free text list | Anatomical locations. `[]` is valid (e.g., cardiomegaly has no location). Typically 0–3 items. |
| `degree` | array of strings | Free text list | Qualifiers, severity, size descriptors. `[]` is valid (~70% of entities have empty degree). Examples: ["mild"], ["small"], ["1.2 cm"], ["bilateral"], ["scattered", "small"] |

## 2.3 Critical Schema Properties

- **Flat entity array** — no section nesting. Old schema had `inputs[].output[]` per FINDINGS/IMPRESSION section; new schema has a single `entities[]`.
- **No measurement field** — sizes and measurements go into `degree` (e.g., `["1.2 cm"]`) or are embedded in the observation string.
- **No general/specific split** — old schema separated `general_finding` (category, e.g., "effusion") from `specific_finding` (detail, e.g., "pleural effusion"). New schema uses ONE `observation` field that can be either level of specificity.
- **Compound split pattern** — a model may split a compound entity: GT has `"coronary artery calcification" @ ["left anterior descending artery"]`, pred has `"calcification" @ ["coronary artery", "left anterior descending artery"]`. The synonym_map compound entries handle this.
- **Dedup key**: `(observation.lower(), observation_presence.lower(), frozenset(location_normalized))` — used to remove duplicate entities that appear in both FINDINGS and IMPRESSION sections.

## 2.4 GT Datasets Available

| File | Entities | Complexity |
|------|----------|-----------|
| gt0.json | 50 | Very complex — chest + abdomen + pelvis + bones. Ventral hernia, SBO. |
| gt1.json | 27 | Complex — paramediastinal mass, spine involvement, atelectasis. |
| gt2.json | 27 | Complex — SVC thrombus, aortic dissection, bilateral pleural effusions. |
| gt3.json | 23 | Moderate — aortic dissection, pulmonary nodules, diverticular disease. |
| gt4.json | 29 | Moderate — metastatic liver disease, ground glass opacities, degenerative. |
| gt5.json | 10 | Stress-test — coronary artery calc, bilateral effusion, hepatomegaly, splenomegaly, lymphadenopathy, pneumothorax, pericardial effusion, atelectasis, nodule, lytic lesions |
| gt6.json | 8  | Unit test — same entities as gt5 subset, used to validate code changes |

---

# 3. FILE-BY-FILE ARCHITECTURE

## 3.1 `ulti_comp_mainn.py` — Main Orchestrator

**Role:** Entry point. Reads config, loads GT + sample files, calls the evaluator, aggregates results, prints the comparison table and SUMMARY.txt.

**Key components:**

```python
# USER CONFIG (top of file — only section the user edits)
class UserConfig:
    DATA_DIR = "./new_schema_v5/chest_ct/5/"   # which GT folder
    GT_FILENAME = "gt5.json"
    # ... LLM toggle, SapBERT model path, etc.
```

**Critical constants:**
```python
FIELD_WEIGHTS = {
    'observation': 0.45,   # Merged general+specific (was 0.25+0.20 in old 6-field schema)
    'location':    0.35,   # Raised from 0.30
    'degree':      0.20,   # Raised from 0.15
    # observation_presence: NOT in weights — used as a gate only (see UCM-3)
    # measurement: REMOVED — does not exist in new schema
}

LOCATION_ABBREVIATIONS = {
    'lad': 'left anterior descending artery',   # FIX R-1 APPLIED: was 'left anterior descending'
    'rca': 'right coronary artery',
    'lcx': 'left circumflex artery',            # FIX R-1 APPLIED: was 'left circumflex'
    'rll': 'right lower lobe', 'rul': 'right upper lobe',
    'rml': 'right middle lobe', 'lll': 'left lower lobe',
    'lul': 'left upper lobe', 'svc': 'superior vena cava',
    'ivc': 'inferior vena cava', 'pa': 'pulmonary artery',
    'mpa': 'main pulmonary artery', 'rpa': 'right pulmonary artery',
    'lpa': 'left pulmonary artery', 'lmsb': 'left mainstem bronchus',
    'rmsb': 'right mainstem bronchus',
}

LOC_NORMMAP = {
    'left lung base': 'left base', 'right lung base': 'right base',
    'bilateral lungs': 'bilateral', 'both lungs': 'bilateral',
    'mediastinum': 'mediastinal',
    'osseous structures': 'bones', 'visualized bones': 'bones',
    'bony structures': 'bones',
    'pleural space': 'pleural', 'pleural cavity': 'pleural',
    'pericardial space': 'pericardial',
}
```

**Field scoring pipeline** (inside `_compute_field_scores()`):
1. Calls `_compare_observation(gt_obs, pred_obs, gt_locs, pred_locs)` → returns `(float, Optional[int])`
   - The `Optional[int]` is `consumed_pred_loc_idx` — the index of a pred location token absorbed by a compound match
2. If compound fired (consumed_idx is not None), strip that index from pred_locs before location scoring
3. Calls `_compare_location(gt_locs_clean, pred_locs_for_scoring)` → float
4. Calls `_compare_degree(gt_deg, pred_deg)` → float
5. Applies presence gate: if observation_presence contradicts GT presence → `gated_score = 0.0`
6. Weighted sum: `FL = 0.45×obs + 0.35×loc + 0.20×deg` (before gate)
7. If gated → `FL_gated = FL × (1 - contradiction_penalty)` where penalty=1.0 for hard contradictions

## 3.2 `entity_level_evaluator.py` — Matching Engine

**Role:** Loads GT and pred entity lists, computes pairwise similarity, applies Hungarian matching, labels each pair as TP/FP/FN, detects contradictions, calls field scorer for TP pairs.

**Key methods:**

| Method | Role |
|--------|------|
| `flatten_entities(schema)` | Extracts `entities[]` from JSON. Handles deduplication. |
| `calculate_entity_similarity(gt, pred)` | Computes overall similarity for one GT↔pred pair for MATCHING purposes (not field scoring). Uses SapBERT only — no synonym map here. |
| `match_entities(gt_list, pred_list)` | Hungarian algorithm. Builds n×m score matrix, finds globally optimal assignment. Returns TP pairs, FP list, FN list. |
| `compare_location(loc1, loc2)` | Used inside matching. Simple semantic Jaccard via SapBERT. |
| `check_contradiction(gt_entity, pred_entity)` | Returns True if observation_presence values are hard contradictions. |
| `semantic_similarity(t1, t2, model)` | Calls SapBERT or other model, returns cosine similarity. |

**CRITICAL: matching vs. field scoring are SEPARATE:**
- `calculate_entity_similarity()` — used during Hungarian matching only. Uses PURE SapBERT. Does NOT use synonym_map. This is intentional (Action E-2 reverted). The matcher must be broad enough to pair synonymous observations, but should not double-count synonym scores.
- `_compare_observation()` in `ulti_comp_mainn.py` — used for field scoring of TP pairs. Uses full 4-step pipeline including synonym_map.

## 3.3 `llm_evaluator.py` — Gemini Evaluator

**Role:** For each sample, takes up to 5 TP pairs and sends them to Gemini Pro with a clinical evaluation prompt. Returns a score 0–1 reflecting clinical semantic correctness.

**Important behaviors:**
- Only evaluates TP pairs (not FPs or FNs directly)
- FP/FN impact is captured via Coverage, not LLM
- Evaluates the first N pairs (default N=5) from the TP list
- LLM score can be toggled off via UserConfig (useful when LLM cost is a concern or for pure structural testing)
- LLM scores are typically consistent with field scores but catch nuances the field scorer misses (e.g., "fluid accumulation" used twice for different entities, clinical credibility of compound synonyms)

## 3.4 `rule_based_validator.py` — Schema Consistency

**Role:** Validates internal consistency of pred entities. Does NOT compare to GT. Produces violation list.

**Current rules:**
| Rule ID | What It Checks |
|---------|---------------|
| PRES-001 | Invalid `observation_presence` value (not "present"/"absent"/"uncertain") |
| PRES-002 | Presence contradicts degree context (e.g., degree="none" but presence="present") |
| ANAT-001 | Location token not in known anatomical ontology |
| DEG-001 | Degree value is numerically impossible (negative sizes, etc.) |

**Important:** Rule violations are a SEPARATE output track. They do NOT reduce the composite score directly. The comparison table shows violation count in the "Rules" column. Violations indicate the pred model made schema errors.

## 3.5 `synonym_map.json` — Synonym & Compound Map

**Current state:** 132 entries (as of latest session).

See Section 13 for full details.

## 3.6 `workflows/chest_ct_workflow.py` — Ontology Normalization

**Role:** Pre-processes entities through a thoracic CT ontology. Normalizes location abbreviations, validates anatomical terms. Called during rule-based validation.

**Does NOT affect matching or field scoring directly** — only feeds into ANAT-001 violations.

---

# 4. COMPLETE EVALUATION PIPELINE — END-TO-END FLOW

```
INPUT: gt.json + sample.json (same report, different entity lists)
         ↓
[1] ENTITY EXTRACTION
    flatten_entities(gt)   → gt_entities[]   (after dedup)
    flatten_entities(pred) → pred_entities[] (after dedup)
    dedup key: (obs.lower(), pres.lower(), frozenset(sorted(locs_normalized)))

         ↓
[2] RULE-BASED VALIDATION (on pred only)
    rule_based_validator.validate(pred_entities)
    → violations[] with rule IDs, severities
    → ontology_coverage % (how many pred obs are in known ontology)
    This runs INDEPENDENTLY of GT comparison.

         ↓
[3] HUNGARIAN MATCHING
    For each GT entity i and pred entity j:
        score[i][j] = calculate_entity_similarity(gt[i], pred[j])
    Apply scipy.optimize.linear_sum_assignment(-score_matrix)
    For each assignment (i,j): if score[i][j] > 0.65 → TP pair
    Unmatched GT → FN | Unmatched pred → FP
    → tp_pairs[], fp_list[], fn_list[]

         ↓
[4] STRUCTURAL METRICS
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2×P×R / (P + R)
    Coverage  = TP / max(|GT|, |Pred|)

         ↓
[5] FIELD-LEVEL SCORING (for TP pairs only)
    For each (gt_entity, pred_entity) in tp_pairs:
        obs_score, consumed_idx = _compare_observation(gt_obs, pred_obs, gt_locs, pred_locs)
        pred_locs_for_scoring   = remove(pred_locs, consumed_idx)  # if compound fired
        loc_score               = _compare_location(gt_locs, pred_locs_for_scoring)
        deg_score               = _compare_degree(gt_deg, pred_deg)
        
        # Presence gate
        if contradiction(gt_presence, pred_presence):
            gated_score = 0.0
        else:
            FL = 0.45×obs + 0.35×loc + 0.20×deg
            gated_score = FL  # × presence_agreement_factor if partial
        
        pair_scores.append(gated_score)
    
    FL_mean     = mean(pair_scores)
    FL_adjusted = FL_mean × Coverage

         ↓
[6] LLM EVALUATION (optional)
    Select up to 5 TP pairs
    Send to Gemini Pro with clinical prompt
    → llm_score ∈ [0, 1]

         ↓
[7] COMPOSITE SCORE
    if LLM enabled:
        Composite = FL_adjusted × LLM_score
    else:
        Composite = FL_adjusted

         ↓
[8] OUTPUT
    Per-sample: F1, FL_mean, Coverage, FL_adj, LLM, Composite, Rules, Contradictions
    SUMMARY.txt with comparison table and aggregate statistics
```

---

# 5. EMBEDDING & SIMILARITY INFRASTRUCTURE

## 5.1 SapBERT

**Model:** `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`

**Why SapBERT over generic BERT/PubMedBERT:**
- Trained specifically on biomedical entity linking via UMLS
- Encodes "effusion" and "pleural fluid" much closer in embedding space than PubMedBERT
- Tested against PubMedBERT, BioWordVec, ClinicalBERT — SapBERT consistently outperformed for medical entity similarity

**Embedding computation:**
```python
# Tokenize, pass through model, mean-pool the token embeddings
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=64)
with torch.no_grad():
    outputs = model(**inputs)
embedding = outputs.last_hidden_state.mean(dim=1)
```

**Cosine similarity** between embeddings → raw similarity score.

**Caching:** Embeddings are cached by text string to avoid recomputation across pairs.

## 5.2 SapBERT Thresholds

Two threshold contexts:

**In matching (`calculate_entity_similarity`):**
- Overall entity similarity threshold: `> 0.65` → TP pair
- Location sub-comparison: `> 0.70` for individual tokens
- These thresholds were empirically validated across GT0–GT4

**In field scoring (`_compare_observation` Step 4):**
- `> 0.85` → 1.0 (effectively exact)
- `0.70–0.85` → partial credit (the raw sim value)
- `< 0.70` → 0.0 (unrelated)

## 5.3 What Was Tried and Rejected for Embeddings

| Approach | Result | Why Rejected |
|---------|--------|-------------|
| PubMedBERT | Dynamic range 0.116 | Too similar for all medical terms — couldn't distinguish good from bad |
| BioWordVec | Good single-token, poor multi-token | Fails on compound phrases |
| ClinicalBERT | Slightly worse than SapBERT | Trained on notes, not entity linking |
| TF-IDF + cosine | Very poor for medical synonyms | Lexical only |
| Soft F1 over token overlap | Required exact substring sharing | "hepatic enlargement" vs "hepatomegaly" → 0 |
| Exact match only | Too strict | "RLL" ≠ "right lower lobe" → penalized |

## 5.4 Embedding Usage Scope

SapBERT is used in:
1. `calculate_entity_similarity()` — building the n×m score matrix for Hungarian matching
2. `_compare_location_terms()` — comparing individual location tokens
3. `_compare_observation()` Step 4 — SapBERT fallback when no synonym/compound fires

SapBERT is NOT used for:
- `observation_presence` comparison — exact string match only (3 values)
- `degree` comparison — exact match after normalization
- Synonym map lookups — those are dictionary lookups

---

# 6. OBSERVATION COMPARISON: THE FULL 4-STEP PIPELINE

This is the most complex component. Located in `ulti_comp_mainn.py` as `_compare_observation()`.

## 6.1 Function Signature and Return Type

```python
def _compare_observation(self,
                         gt_obs: str, pred_obs: str,
                         gt_locs: list = None,
                         pred_locs: list = None) -> Tuple[float, Optional[int]]:
```

**Returns:** `(score: float, consumed_pred_loc_idx: Optional[int])`

`consumed_pred_loc_idx` is the index into `pred_locs` of a location token that was absorbed by a compound match. The caller (`_compute_field_scores`) must remove this index from pred_locs before passing to location scoring, preventing double-penalization.

## 6.2 The 4 Steps

### Step 1: Exact String Match
```python
if gt_clean == pred_clean:
    return 1.0, None
if not gt_clean or not pred_clean:
    return 0.0, None
```

### Step 2: Synonym Map Resolution
```python
gt_canonical,   gt_conf   = _resolve_synonym(gt_clean)
pred_canonical, pred_conf = _resolve_synonym(pred_clean)

if gt_canonical == pred_canonical:
    # Both resolve to same canonical term.
    # BUT: check if compound also fires — prefer compound (allows location consumption).
    for ckey, idx in _build_compound_keys(pred_clean, pred_locs):
        entry = SYNONYM_MAP.get(ckey)
        if entry and entry['canonical'] == gt_canonical:
            return min(entry['confidence'], gt_conf), idx  # compound wins, idx consumed
    return min(gt_conf, pred_conf), None   # plain synonym
```

**Why compound priority in Step 2:** When pred observation `"calcified plaque"` is at locations `["coronary artery", "lad"]`:
- Plain synonym: `map["calcified plaque"] → "coronary artery calcification"` conf=0.80
- Compound: `map["calcified plaque|coronary artery"] → "coronary artery calcification"` conf=0.90

Without compound priority, synonym fires first → returns `(0.80, None)` → "coronary artery" stays in pred_locs → location scorer penalizes the extra token → `loc=0.67` instead of `1.0`.

With compound priority, the compound path fires → returns `(0.90, 0)` → "coronary artery" is consumed → location scorer sees only `["lad"]` → expands to `["left anterior descending artery"]` = GT → `loc=1.0`.

### Step 3A: DIR-A Compound (GT-side)
```python
for compound_key, _loc_idx in _build_compound_keys(gt_clean, gt_locs):
    entry = SYNONYM_MAP.get(compound_key)
    if entry:
        cc, ccf = entry['canonical'], entry['confidence']
        if cc == pred_canonical:
            return min(ccf, pred_conf), None    # GT loc consumed — no pred idx
        resolved, rc = _resolve_synonym(cc)
        if resolved == pred_canonical:
            return min(ccf, rc, pred_conf), None
```

DIR-A: used when the GT observation is generic and GT location makes it specific. Example:
- GT: `"lymphadenopathy"` @ `["mediastinal"]` — compound key = `"lymphadenopathy|mediastinal"`
- Pred: `"mediastinal lymph node enlargement"` @ `[]`
- compound fires → score returned — GT loc consumed, no pred loc consumed (None)

### Step 3B: DIR-B Compound (Pred-side)
```python
for compound_key, loc_idx in _build_compound_keys(pred_clean, pred_locs):
    entry = SYNONYM_MAP.get(compound_key)
    if entry:
        cc, ccf = entry['canonical'], entry['confidence']
        if cc == gt_canonical:
            return min(ccf, gt_conf), loc_idx   # pred loc consumed → idx returned
        resolved, rc = _resolve_synonym(cc)
        if resolved == gt_canonical:
            return min(ccf, rc, gt_conf), loc_idx
```

DIR-B: used for compound split patterns. Example:
- GT: `"coronary artery calcification"` @ `["left anterior descending artery"]`
- Pred: `"calcification"` @ `["coronary artery", "left anterior descending artery"]`
- compound key = `"calcification|coronary artery"` → fires → returns `(1.0, 0)` → "coronary artery" consumed

### Step 4: SapBERT Fallback
```python
sim = self._sapbert_similarity(gt_clean, pred_clean)
if sim > 0.85:
    return 1.0, None
elif sim > 0.70:
    return sim, None     # partial credit
else:
    return 0.0, None
```

**Note:** `gt_clean` and `pred_clean` are used here, NOT `gt_term`/`pred_term`. This was a bug in earlier versions.

## 6.3 `_build_compound_keys(obs, locs)` Helper
```python
def _build_compound_keys(obs: str, locs: list) -> List[Tuple[str, int]]:
    o = obs.lower().strip()
    results = []
    for i, loc in enumerate(locs):
        lc = normalize_loc_token(str(loc).lower().strip())
        results.append((f"{o}|{lc}", i))
        results.append((f"{lc}|{o}", i))
    return results
```

Both key orders are tried to handle the entry format in synonym_map.json.

## 6.4 `_resolve_synonym(term)` Helper
```python
def _resolve_synonym(term: str) -> Tuple[str, float]:
    entry = SYNONYM_MAP.get(term.lower().strip())
    if entry and isinstance(entry, dict) and 'canonical' in entry:
        return entry['canonical'], entry['confidence']
    return term.lower().strip(), 1.0  # no map entry → term is its own canonical
```

When a term has no map entry, it returns `(term, 1.0)`. This means gt_canonical == pred_canonical only when BOTH sides either map to the same canonical OR are identical strings. Two unknown terms that are different will not match in Step 2 (they will have the same canonical only if they're identical strings).

---

# 7. LOCATION COMPARISON

## 7.1 `_compare_location_terms()` in `ulti_comp_mainn.py`

Used in field scoring for TP pairs.

```python
def _compare_location_terms(self, gt_locs: list, pred_locs: list) -> float:
    if not gt_locs and not pred_locs:
        return 1.0
    
    if not gt_locs:
        # FIX R-2 APPLIED: GT empty, pred adds specificity → partial credit
        return 0.7   # was 0.0 before fix
    
    if not pred_locs:
        return 0.0   # pred missing GT-specified location → genuine loss
    
    # Normalize both sides
    gt_n   = [normalize_loc_token(l) for l in gt_locs]
    pred_n = [normalize_loc_token(l) for l in pred_locs]
    
    # Greedy matching with SapBERT
    total_sim = 0.0
    matched = set()
    for gl in gt_n:
        best_s, best_j = 0.0, -1
        for j, pl in enumerate(pred_n):
            if j in matched: continue
            s = 1.0 if gl == pl else self._sapbert_similarity(gl, pl)
            # PENDING FIX C-3: anatomical region fallback
            # if s < 0.70:
            #     if _region_key(gl) and _region_key(gl) == _region_key(pl):
            #         s = 0.80
            if s > best_s:
                best_s, best_j = s, j
        if best_s > 0.70:
            total_sim += best_s
            matched.add(best_j)
    
    prec = total_sim / len(gt_n)
    rec  = total_sim / len(pred_n)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)   # F1-style harmonic mean
```

## 7.2 `normalize_loc_token()` Pipeline
```python
def normalize_loc_token(t: str) -> str:
    t = LOCATION_ABBREVIATIONS.get(t.lower().strip(), t.lower().strip())  # expand abbrev
    if t in LOC_NORMMAP:
        return LOC_NORMMAP[t]
    for suffix in [' space', ' cavity', ' region', ' area', ' structures',
                   ' territory', ' field', ' zone', ' aspect']:
        if t.endswith(suffix):
            t = t[:-len(suffix)].strip()
    return t
```

Order matters: abbreviation expansion first, then explicit normmap, then suffix stripping.

## 7.3 `compare_location()` in `entity_level_evaluator.py`

Used during MATCHING (not field scoring). Simpler version:
```python
def compare_location(self, loc1: list, loc2: list) -> float:
    if not loc1 and not loc2:
        return 1.0
    if not loc1:
        return 0.7   # FIX R-2 APPLIED (same fix as field scorer)
    if not loc2:
        return 0.0
    # Same greedy SapBERT-based F1 logic as above
```

---

# 8. DEGREE, PRESENCE & COVERAGE

## 8.1 Degree Comparison

```python
def _compare_degree(self, gt_deg: list, pred_deg: list) -> float:
    if not gt_deg and not pred_deg:
        return 1.0
    if not gt_deg or not pred_deg:
        return 0.0
    
    # PENDING FIX C-2: strip prefixes before comparison
    # def _strip_deg_prefix(d):
    #     d = d.lower().strip()
    #     for pfx in ("up to ", "at least ", "approximately ", "about ", "at most "):
    #         if d.startswith(pfx): return d[len(pfx):].strip()
    #     return d
    
    gt_set   = set(d.lower().strip() for d in gt_deg)
    pred_set = set(d.lower().strip() for d in pred_deg)
    
    # Check for antonyms first (semantic opposites)
    for g in gt_set:
        for p in pred_set:
            if (g, p) in DEGREE_ANTONYMS or (p, g) in DEGREE_ANTONYMS:
                return 0.0
    
    # Jaccard overlap
    intersection = gt_set & pred_set
    union = gt_set | pred_set
    jaccard = len(intersection) / len(union) if union else 0.0
    
    if jaccard > 0:
        return jaccard
    
    # SapBERT fallback for non-matching degree terms
    # (e.g., "mild" vs "moderate" → partial credit ~0.7)
    sims = []
    for g in gt_set:
        for p in pred_set:
            sims.append(self._sapbert_similarity(g, p))
    avg_sim = max(sims) if sims else 0.0
    return avg_sim if avg_sim > 0.70 else 0.0
```

**DEGREE_ANTONYMS examples:** `{("small", "large"), ("absent", "present"), ("unilateral", "bilateral"), ("partial", "complete"), ...}`

**PENDING BUG:** "up to 1.2 cm" ≠ "1.2 cm" because prefix "up to" not stripped. Fix C-2 is written but not yet applied.

## 8.2 Presence Gate

`observation_presence` is NOT in `FIELD_WEIGHTS`. It is a categorical gate:

```python
HARD_CONTRADICTIONS = {
    ('present', 'absent'), ('absent', 'present'),
    ('present', 'uncertain'), ('uncertain', 'present'),
}

if (gt_presence.lower(), pred_presence.lower()) in HARD_CONTRADICTIONS:
    gated_score = 0.0   # Zero out FL_mean for this pair
else:
    gated_score = FL_weighted_mean
```

This means: if the model says "absent" when GT says "present", ALL field scores for that pair are zeroed. The entity was matched (it formed a TP via Hungarian, since the observation name matched) but its field quality is 0.

**Why gate instead of penalize:** Presence errors are clinical safety issues. A model that identifies the right entity but says it's absent when it's present has made a fundamental error that cannot be partially credited.

## 8.3 Coverage

```python
Coverage = TP / max(len(gt_entities), len(pred_entities))
```

**Note:** This differs from standard Recall `(TP / GT_count)`. Coverage uses max(GT, Pred) in the denominator, so it penalizes BOTH over-extraction (FPs) and under-extraction (FNs).

## 8.4 FL Adjusted

```python
FL_adjusted = FL_mean × Coverage
```

This single multiplication combines field quality with entity completeness. It was found to be the most predictive single metric (before LLM) when compared to human rankings.

---

# 9. HUNGARIAN MATCHING (ENTITY LEVEL)

## 9.1 Why Hungarian (Not Greedy)

Greedy matching commits to the highest-similarity pair first and cannot undo that assignment. This fails when:
- A compound-split entity has moderate similarity to two GT entities and the "wrong" GT entity wins first
- Synonymous observations compete for the same GT entity

Hungarian algorithm finds the globally optimal assignment (maximizes total similarity across all pairs simultaneously). Complexity O(n³), fast enough for n=10–15 entities.

## 9.2 Implementation

```python
from scipy.optimize import linear_sum_assignment

def match_entities(self, gt_list, pred_list):
    n_gt   = len(gt_list)
    n_pred = len(pred_list)
    
    # Build n_gt × n_pred similarity matrix
    score_matrix = np.zeros((n_gt, n_pred))
    for i, gt in enumerate(gt_list):
        for j, pred in enumerate(pred_list):
            score_matrix[i][j] = self.calculate_entity_similarity(gt, pred)
    
    # Hungarian: minimize -score = maximize score
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    
    tp_pairs, fp_list, fn_list = [], [], []
    matched_gt, matched_pred = set(), set()
    
    for i, j in zip(row_ind, col_ind):
        if score_matrix[i][j] > 0.65:   # threshold
            tp_pairs.append((gt_list[i], pred_list[j], score_matrix[i][j]))
            matched_gt.add(i)
            matched_pred.add(j)
    
    fn_list = [gt_list[i]   for i in range(n_gt)   if i not in matched_gt]
    fp_list = [pred_list[j] for j in range(n_pred) if j not in matched_pred]
    
    return tp_pairs, fp_list, fn_list
```

## 9.3 `calculate_entity_similarity()` — For Matching Only

Uses **pure SapBERT, no synonym map**. This is Action E-2 (the revert from a prior buggy version that injected synonym scores into the matcher).

```python
def calculate_entity_similarity(self, gt_entity, pred_entity) -> float:
    gt_obs   = gt_entity.get('observation', '')
    pred_obs = pred_entity.get('observation', '')
    gt_loc   = gt_entity.get('location', [])
    pred_loc = pred_entity.get('location', [])
    gt_pres  = gt_entity.get('observation_presence', '')
    pred_pres = pred_entity.get('observation_presence', '')
    
    # SapBERT similarity between observation strings
    finding_sim  = self.semantic_similarity(gt_obs, pred_obs, 'sapbert')
    location_sim = self.compare_location(gt_loc, pred_loc)
    presence_sim = 1.0 if gt_pres.lower() == pred_pres.lower() else 0.0
    
    # Weighted combination for matching score
    overall = (0.40 * finding_sim +
               0.30 * location_sim +
               0.20 * presence_sim +
               0.10 * 1.0)   # base score
    return overall
```

**Weights rationale:**
- 0.40 on finding: most important for entity identity
- 0.30 on location: prevents "cardiomegaly" from matching "pneumothorax" just because both have empty location
- 0.20 on presence: contradictions can still be matched (necessary for contradiction detection to work)
- 0.10 base: ensures minimum matching tendency for partial cases

**Why not use synonym map here:** If the matcher used synonym scores, it would see artificially high similarity for synonym pairs and assign them confidently — but then the field scorer would ALSO apply synonym scoring to the same pair → double-counting. The matcher needs raw semantic distance; the field scorer applies the calibrated confidence weights.

---

# 10. LLM EVALUATOR

## 10.1 Role in the Pipeline

LLM is the final multiplicative factor: `Composite = FL_adjusted × LLM`

It acts as a clinical sanity check. The field scorer can be fooled by technically matching tokens that are clinically wrong. The LLM evaluator catches:
- "fluid accumulation" used for both effusion AND pericardial effusion in the same pred — ambiguous reuse
- Degree imprecision ("severe" vs "mild") that gets partial SapBERT credit but is clinically significant
- Compound splits where the observation makes clinical sense but the overall entity is still debatable
- Extra FP entities that are clinically in the report but not in GT (signals over-extraction)

## 10.2 What LLM Evaluates

For up to 5 TP pairs, the prompt includes:
- The original radiology report text
- Each TP pair as (GT entity, Pred entity)
- A rubric asking: "For each pair, score 0–1 reflecting how clinically equivalent these two entity descriptions are"

LLM score = mean across the evaluated pairs.

## 10.3 What LLM Does NOT See

- FP entities (hallucinations not in GT)
- FN entities (GT entities the model missed)

These are captured by Coverage, not LLM. This is intentional — LLM evaluates quality of matched pairs, Coverage evaluates completeness.

## 10.4 LLM Costs and Toggle

LLM calls are expensive. Can be disabled in UserConfig:
```python
USE_LLM = False   # When False: Composite = FL_adjusted (no LLM multiplier)
```

When disabled, the system is still highly correlated with human ranking via FL_adjusted alone (Pearson r ~0.90+) but loses the clinical nuance check.

---

# 11. RULE-BASED VALIDATOR

## 11.1 Scope

Validates PRED entities only for internal schema consistency. Never compares to GT. Catches cases where the model produced structurally invalid output.

## 11.2 Rules

**PRES-001 (Error):** `observation_presence` is not "present", "absent", or "uncertain".
```python
if pred_pres not in {"present", "absent", "uncertain"}:
    violations.append(RuleViolation("PRES-001", SEVERITY_ERROR, ...))
```

**PRES-002 (Error):** Presence says "absent" but degree contains positive qualifiers.
```python
# e.g., observation_presence="absent", degree=["moderate", "bilateral"]
# The model is simultaneously saying it's absent and describing it as moderate
```

**ANAT-001 (Warning):** Location token not in known thoracic CT anatomical ontology.
```python
# "banana" in location → warning
# "mediastinal" → known → no warning
```

**DEG-001 (Info):** Unusual degree value patterns (overly long strings, special characters).

## 11.3 Violations Do Not Affect Composite

Violations are shown in the comparison table "Rules" column as a count. They do NOT subtract from Composite score. Rationale: the evaluator's job is to score quality of matched entities; structural rule violations are a separate quality dimension reported independently.

---

# 12. COMPOSITE SCORE CONSTRUCTION

## 12.1 Formula

```
FL_mean     = mean(FL_gated[tp_i])   for all TP pairs
Coverage    = TP / max(|GT|, |Pred|)
FL_adjusted = FL_mean × Coverage

if LLM enabled:
    Composite = FL_adjusted × LLM_score
else:
    Composite = FL_adjusted
```

## 12.2 Why This Formula Works

The multiplicative structure ensures:
- A sample with FL_adj=0.9 but LLM=0.4 gets Composite=0.36 (LLM caught clinical issues)
- A sample with perfect entity matching but 50% coverage gets Composite=0.5 max (can't hide FNs/FPs)
- A sample with all presence contradictions: FL_gated→0 → FL_mean→0 → FL_adj→0 → Composite→0

## 12.3 Alternative Formulas Tried and Rejected

| Formula | Result | Problem |
|---------|--------|---------|
| Pure F1 | Insensitive to field quality | Two samples with F1=0.7 could be very different quality |
| PubMedBERT FL only | Range 0.80–0.92 | No dynamic range — all samples look similar |
| F1 × FL | Correlation 0.65 with human | F1 amplification wrong direction |
| Additive: 0.4×F1 + 0.3×FL + 0.3×LLM | 0.72 corr | Doesn't penalize enough for contradictions |
| Coverage × LLM (no FL) | 0.70 corr | Loses field-level information |
| FL_adj × LLM (current) | ~0.96 corr with human | Best found empirically across GT0–GT4 |

---

# 13. SYNONYM_MAP.JSON — STRUCTURE, LOGIC, CURRENT STATE

## 13.1 File Structure

```json
{
  "_metadata": {
    "version": "2.0",
    "total_entries": 132,
    "last_updated": "2026-02-27"
  },
  "effusion": {
    "canonical": "effusion",
    "confidence": 1.0
  },
  "pleural fluid": {
    "canonical": "effusion",
    "confidence": 0.95
  },
  "fluid accumulation": {
    "canonical": "effusion",
    "confidence": 0.95
  },
  "calcification|coronary artery": {
    "canonical": "coronary artery calcification",
    "confidence": 1.0
  },
  ...
}
```

## 13.2 Entry Types

**Plain synonym entries:**
```json
"hepatic enlargement": {"canonical": "hepatomegaly", "confidence": 0.95}
```
The key is a single observation term. The canonical is what it maps to. confidence reflects how precisely this synonym refers to the canonical (1.0 = exact synonym, 0.70 = approximate).

**Compound entries:**
```json
"calcification|coronary artery": {"canonical": "coronary artery calcification", "confidence": 1.0}
"coronary artery|calcification": {"canonical": "coronary artery calcification", "confidence": 1.0}
```
The key is `obs|location` or `location|obs`. Both directions are stored. Used when an entity is split: `"calcification" @ ["coronary artery", "lad"]` → compound key fires → resolves to "coronary artery calcification".

## 13.3 Confidence Values

| Confidence | Meaning | Example |
|-----------|---------|---------|
| 1.00 | Exact synonym or canonical | "effusion" itself, "adenopathy"→"lymphadenopathy" |
| 0.95 | Very close synonym | "pleural fluid"→"effusion", "pericardial fluid"→"pericardial effusion" |
| 0.90 | Close synonym with minor nuance | "fluid collection"→"effusion", "bilateral pleural collection"→"effusion" |
| 0.85 | Moderate synonym | "collapse"→"atelectasis", "volume loss"→"atelectasis" |
| 0.80 | Generic term mapping to specific | "calcified plaque"→"coronary artery calcification", "arterial calcification"→"calcification" |
| 0.70 | Weakest meaningful synonym | "small opacity"→"nodule", "osseous metastases"→"lytic lesions" |

## 13.4 Current Map Coverage (132 entries as of latest session)

**Observation synonyms (plain):** ~80 entries
Key examples:
- `pleural fluid`, `fluid accumulation` → `effusion`
- `fluid collection` → `effusion`
- `cardiac enlargement`, `cardiac silhouette enlargement` → `cardiomegaly`
- `hepatic enlargement`, `enlarged hepatic silhouette` → `hepatomegaly`
- `splenic enlargement` → `splenomegaly`
- `volume loss`, `collapse`, `linear atelectasis` → `atelectasis`
- `pulmonary nodule`, `small opacity` → `nodule`
- `adenopathy`, `enlarged lymph nodes`, `mediastinal lymph node enlargement`, `lymph node enlargement` → `lymphadenopathy`
- `pericardial fluid` → `pericardial effusion`
- `lytic bone lesions` → `lytic lesions`
- `osseous metastases` → `lytic lesions` (conf=0.70)
- `arterial calcification` → `calcification` (conf=0.80)
- `calcified plaque` → `coronary artery calcification` (conf=0.80)
- `pleural effusion` → `effusion`

**Compound entries (~52 entries, both key orders):**
- `calcification|coronary artery` → `coronary artery calcification` (1.0)
- `enlarged|liver`, `enlargement|liver` → `hepatomegaly` (0.95)
- `enlarged|spleen`, `enlargement|spleen`, `enlarged|splenic`, `enlargement|splenic` → `splenomegaly` (0.95)
- `enlargement|heart`, `enlarged|heart` → `cardiomegaly` (0.95)
- `fluid accumulation|pericardial` → `pericardial effusion` (0.95)
- `calcified plaque|coronary artery` → `coronary artery calcification` (0.90) ← FIX BUG 5.2-A
- `arterial calcification|left anterior descending artery` → `coronary artery calcification` (0.85)
- `arterial calcification|lad` → `coronary artery calcification` (0.85)
- `pleural fluid|bilateral` → `effusion` (0.95) ← supports implicit-location fix

## 13.5 Map Philosophy

The map should contain CORRECT synonym relationships. Adding a term to the map that is clinically ambiguous or wrong would be a false positive generator. For example:
- `"pleural thickening"` → `"effusion"`: DO NOT ADD. These are different pathologies.
- `"hepatic steatosis"` → `"hepatomegaly"`: DO NOT ADD. Steatosis ≠ enlargement.
- `"normal spleen size"` → `"splenomegaly"`: DO NOT ADD. This is the negation.

When a pred uses the wrong medical term (different pathology), the system SHOULD score it low. The map should only contain real synonyms.

---

# 14. ALL APPLIED CODE CHANGES (CHRONOLOGICAL)

## 14.1 Action E-2 Revert — `entity_level_evaluator.py`

**Problem:** E-2 injected synonym/compound scores into `calculate_entity_similarity()`. This caused the greedy matcher (pre-Hungarian) to assign synonym-boosted pairs first, breaking chains and causing TPs to become FNs in other samples.

**Fix:** `calculate_entity_similarity()` now uses ONLY SapBERT. No synonym_map lookup in the matcher.

```python
# BEFORE (wrong):
# calculate_entity_similarity used synonym map to boost finding_sim

# AFTER (correct):
finding_sim = self.semantic_similarity(gt_obs, pred_obs, 'sapbert')
# No synonym_map. Pure SapBERT cosine similarity only.
```

## 14.2 Hungarian Matching — `entity_level_evaluator.py`

**Problem:** Greedy matching committed to first-best pair and couldn't undo.

**Fix:** `match_entities()` now uses `scipy.optimize.linear_sum_assignment`.

```python
# BEFORE: greedy loop
# AFTER:
from scipy.optimize import linear_sum_assignment
score_matrix = ...  # n_gt × n_pred
row_ind, col_ind = linear_sum_assignment(-score_matrix)
# Then threshold check per assignment
```

**Effect on GT6 test:** sample6.3 went from F1=0.500 (4 TPs out of 8, couldn't see contradictions) to F1=1.000 (8 TPs, all 4 contradictions detected). Composite went from 0.500 to 0.226 (presence gate correctly zeros out contradicting pairs).

## 14.3 Consumed-Location Fix — `ulti_comp_mainn.py`

**Problem:** When compound matching absorbed a location token into the observation score, that token was being penalized AGAIN in the location scorer.

**Fix:** `_compare_observation` returns `(float, Optional[int])` tuple. The int is the consumed pred location index. `_compute_field_scores` unpacks the tuple, removes the consumed index from `pred_locs` before passing to location scorer.

```python
# _compare_observation now returns (score, consumed_idx)
obs_score, consumed_idx = _compare_observation(gt_obs, pred_obs, gt_locs, pred_locs)

# Remove consumed pred location from loc scoring
pred_locs_for_scoring = [l for i, l in enumerate(pred_locs) if i != consumed_idx]
                         if consumed_idx is not None else pred_locs

loc_score = _compare_location(gt_locs, pred_locs_for_scoring)
```

## 14.4 FIX R-1 — `ulti_comp_mainn.py` LOCATION_ABBREVIATIONS

**Problem:** `'lad'` expanded to `'left anterior descending'` (missing " artery"). GT uses `'left anterior descending artery'`. SapBERT similarity between these falls below 0.70 threshold → loc=0.0.

**Fix:**
```python
# BEFORE:
'lad': 'left anterior descending',
'lcx': 'left circumflex',

# AFTER:
'lad': 'left anterior descending artery',
'lcx': 'left circumflex artery',
```

## 14.5 FIX R-2 — `entity_level_evaluator.py` and `ulti_comp_mainn.py`

**Problem:** When GT location is empty but pred has a location token (e.g., GT "cardiomegaly" @ [] vs pred "enlargement" @ ["heart"]), `compare_location([], ["heart"])` returned 0.0 hard zero. This killed the overall entity similarity score below 0.65, preventing the pair from forming a TP entirely.

**Fix:**
```python
# BEFORE:
if not loc1 or not loc2:
    return 0.0

# AFTER:
if not loc1:
    return 0.7   # pred adds specificity, not a mismatch
if not loc2:
    return 0.0   # pred omits GT-specified location → genuine loss
```

**Rationale:** "cardiomegaly" is definitionally cardiac. Pred saying "heart" location is adding explicit information, not contradicting. Return 0.7 (not 1.0 because we can't verify pred's location is correct; not 0.0 because it's not a mismatch).

---

# 15. OUTSTANDING BUGS & FIXES PENDING

## BUG C-1: Synonym fires before compound — blocks location consumption

**Affected:** sample5.2 E1 (`calcified plaque @ [coronary artery, lad]`)

**Problem:** When pred observation has BOTH a plain synonym entry AND a compound entry with a location token, Step 2 fires first (plain synonym) and returns without consuming the location token. The location token then stays in pred_locs and is penalized in the location scorer.

**Status:** Compound priority logic is written and described. Map entry `"calcified plaque|coronary artery"` (conf=0.90 > plain synonym conf=0.80) has been added to synonym_map.json. The code change in Step 2 of `_compare_observation` is described but NOT YET APPLIED to the actual .py file.

**Code change needed (in `_compare_observation` Step 2):**
```python
# FIND:
if gt_canonical == pred_canonical:
    return min(gt_conf, pred_conf), None

# REPLACE WITH:
if gt_canonical == pred_canonical:
    # Check if compound also fires — prefer compound to allow location consumption
    for ckey, idx in _build_compound_keys(pred_clean, pred_locs):
        entry = SYNONYM_MAP.get(ckey)
        if entry and entry['canonical'] == gt_canonical:
            return min(entry['confidence'], gt_conf), idx
    return min(gt_conf, pred_conf), None
```

**Expected impact:** sample5.2 E1 loc 0.67→1.0. Overall sample5.2 composite ~0.867→~0.950.

---

## BUG C-2: Degree prefix "up to" not stripped (U-4 not applied)

**Affected:** sample5.0 E5, sample5.2 E5 (`"up to 1.2 cm"` vs `"1.2 cm"`)

**Problem:** GT degree `["1.2 cm"]`, pred degree `["up to 1.2 cm"]`. No prefix stripping → exact set match fails → degree score=0.0.

**Status:** NOT YET APPLIED to .py file.

**Code change needed (in `_compare_degree`):**
```python
# Add this helper and apply before set construction:
def _strip_deg_prefix(d: str) -> str:
    d = d.lower().strip()
    for pfx in ("up to ", "at least ", "approximately ", "about ", "at most ", "no more than "):
        if d.startswith(pfx):
            return d[len(pfx):].strip()
    return d

gt_set   = set(_strip_deg_prefix(str(d)) for d in gt_deg)
pred_set = set(_strip_deg_prefix(str(d)) for d in pred_deg)
```

---

## BUG C-3: "left lower lobe" ≠ "left base" after normalize_loc

**Affected:** sample5.0 E8 (`"collapse @ [left lower lobe]"` vs GT `"atelectasis @ [left lung base]"`)

**Problem:** `normalize_loc("left lung base")` → `"left base"`. `normalize_loc("left lower lobe")` → `"left lower lobe"` (no rule applies). SapBERT("left base", "left lower lobe") is likely < 0.70 threshold → loc=0.0.

Clinically: "left lower lobe" and "left lung base" are the same anatomical region in CT reporting.

**Status:** NOT YET APPLIED. Two options:

Option A — LOC_NORMMAP addition:
```python
LOC_NORMMAP['left lower lobe'] = 'left base'
```
Risk: "left lower lobe" is not always the same as "left base" in all contexts.

Option B — Anatomical region fallback in `_compare_location_terms` (recommended):
```python
# In _compare_location_terms, after SapBERT sim computed:
if sim < 0.70:
    def _region_key(t):
        side  = 'left' if 'left' in t else ('right' if 'right' in t else '')
        level = 'lower' if ('lower' in t or 'base' in t) else \
                ('upper' if 'upper' in t else \
                ('middle' if 'middle' in t else ''))
        return (side, level) if side and level else None
    if _region_key(gt_term) and _region_key(gt_term) == _region_key(pred_term):
        sim = 0.80
```

**NOTE: This block belongs in `_compare_location_terms`, NOT in `_compare_observation`.** Earlier versions had it misplaced.

---

## ARCHITECTURAL GAP A-1: Location absorbed into observation synonym

**Affected:** sample5.2 E2, sample5.3 E2, E5, E10

**Problem:** When a pred observation ENCODES a location word within the observation string (e.g., `"pleural fluid"`, `"mediastinal lymph node enlargement"`, `"bilateral pleural collection"`), and the synonym resolves correctly, the location word in the observation string is semantically consumed. But GT lists that same word in its `location` array. Current scorer penalizes pred for not also listing it in pred_locs.

**Example:**
```
GT  obs: "effusion"  @ ["pleural", "bilateral"]
Pred obs: "pleural fluid" @ ["bilateral"]
SYNONYM: "pleural fluid" → "effusion" (0.95) ✓  obs score OK
LOC: GT has ["pleural", "bilateral"], pred has ["bilateral"] → "pleural" unmatched → loc=0.67
PROBLEM: pred encoded "pleural" in the obs string → should be exempt from loc penalty
```

**Status:** Conceptually clear, not yet implemented. Complex fix needed.

**Proposed implementation:**
In `_compute_field_scores`, after synonym match:
```python
# Detect which GT location tokens are substrings of pred observation surface form
# (only when synonym conf >= 0.85 to avoid over-exemption)
if obs_conf >= 0.85:
    implicit_gt_locs = set(
        l for l in gt_locs_normalized
        if l in pred_obs.lower() or pred_obs.lower() in l
    )
    gt_locs_for_scoring = [l for l in gt_locs_normalized
                           if l not in implicit_gt_locs]
else:
    gt_locs_for_scoring = gt_locs_normalized
loc_score = _compare_location(gt_locs_for_scoring, pred_locs_for_scoring)
```

**Expected impact:** sample5.3 E2 loc 0.0→1.0, E5 loc 0.0→1.0, E10 loc 0.0→1.0. sample5.3 composite ~0.613→~0.80+.

---

# 16. WHAT WAS TRIED AND REJECTED

## 16.1 Embedding Models (See Section 5.3)

PubMedBERT, BioWordVec, ClinicalBERT all rejected. SapBERT was best by large margin for medical entity linking.

## 16.2 Greedy Matching (Replaced by Hungarian)

Greedy was the initial implementation. Rejected because it failed on compound-split entities and contradicting entity pairs (could not detect contradictions). See Action 14.2.

## 16.3 Synonym Scores in the Matcher (E-2 Revert)

Early version (E-2) injected synonym_map confidence scores into `calculate_entity_similarity()`. This raised TP counts for synonym pairs but caused regressions in samples with multiple synonym-able entities (greedy committed to the wrong pairs). Reverted. Matching uses pure SapBERT; field scoring uses synonym_map.

## 16.4 Measurement Field

Old schema had a `measurement` field. Tried carrying it over as a 5th field with weight 0.10. Found that measurements rarely appear in new schema predictions (absorbed into degree), and the field added noise. Removed entirely. New FIELD_WEIGHTS: obs=0.45, loc=0.35, deg=0.20.

## 16.5 General/Specific Split

Old schema split observation into `general_finding` (category) + `specific_finding` (detail). Tried maintaining this split in v3.0 with merged observations by artificially splitting "pleural effusion" → general="effusion", specific="pleural effusion". Created inconsistencies. Abandoned. New schema uses single `observation` field evaluated as-is.

## 16.6 Binary Presence Score in Composite

Tried adding `presence_accuracy = correct_presences / TP_count` as a separate composite component. Rejected because: (a) the gate approach already captures contradictions correctly; (b) adding a presence term double-counted the penalty; (c) the presence gate + coverage combination was already sufficient.

## 16.7 LLM as Primary Metric

Tried using LLM score alone (no FL_adj). Correlation with human: ~0.70. The LLM score is too noisy at the entity-pair level to serve as a sole metric. Works best as a multiplier on FL_adj.

## 16.8 Additive Composite Formula

Tried `0.4×F1 + 0.3×FL_adj + 0.3×LLM`. Correlation with human was 0.72. The additive form doesn't propagate zeros correctly — a sample with all contradictions still got non-zero score from F1 component. Multiplicative `FL_adj × LLM` propagates zeros correctly.

---

# 17. GT DATASET BENCHMARKS

## 17.1 GT0–GT4 (Validated Pre-GT5)

**GT4 (6 samples, 4 GT entities — simpler dataset for early validation):**
```
Ranking: 4.2=4.4(0.975) > 4.0(0.548) > 4.3(0.000) > 4.1=4.5(0.000)
Human:   4.2=4.4 > 4.0 > 4.3 > 4.1=4.5
Result:  PERFECT match ✅
```

**GT0 (4 samples, 26 GT entities after dedup — most complex):**
```
Ranking: 0.0(0.703) > 0.2(0.268) > 0.1(0.001) > 0.3(0.000)
Human:   0.0 > 0.2 > 0.1 > 0.3
Result:  PERFECT match ✅
```

**GT1, GT2, GT3:** All validated with perfect ranking match to human scores (see V3_POST_FIX_AUDIT.txt for details).

## 17.2 GT6 (Unit Test Dataset — 8 entities, 5 samples)

Designed to test specific features in isolation. All 9 checks passing post-fix.

```
Sample  Design                          F1     Composite  Expected
6.0     Perfect (exact copy of GT)      1.000  1.000      Baseline ✅
6.1     Synonyms only                   1.000  0.883      Synonym map ✅
6.2     Compound+normalization          0.875  0.849      Consumed-loc fix ✅
6.3     4 presence contradictions       1.000  0.226      Hungarian + gate ✅
6.4     4 hallucinated entities         0.500  0.500      Coverage penalty ✅

Ranking: 6.0 > 6.1 > 6.2 > 6.4 > 6.3  ← correct clinical severity order ✅
```

**What GT6 validated:**
- Hungarian pairs all 8 contradiction entities (previously only 4 were TP, now all 8 are TP including contradictions)
- Presence gate correctly zeros out 4 contradicting pairs → FL_mean=0.537 for 6.3
- Consumed-location fix: "coronary artery" absorbed → not double-penalized
- LLM correctly distinguishes 6.3 (score=0.420) from perfect samples
- Ranking correctly encodes: Perfect > Synonyms > Compound+Norm > Hallucination > Contradictions

## 17.3 GT5 (Real Stress Test — 10 entities, 5 samples)

More challenging than GT6 because pred entities use creative compound splits, and some terms are not in the synonym map.

```
CURRENT (post-fix, pre-C1/C2/C3):
Ranking: 5.2(0.867) > 5.3(0.613) > 5.0(0.320) > 5.4(0.052) > 5.1(0.008)
Expected: 5.2 > 5.0 > 5.3 > 5.4 > 5.1
Problem: 5.0 is below 5.3
```

**Why 5.0 < 5.3 (detailed):**

Sample 5.0 has 3 stacked issues:
1. E5 degree "up to 1.2 cm" ≠ "1.2 cm" → deg=0 (fix C-2)
2. E8 location "left lower lobe" vs "left base" → loc=0 (fix C-3, SapBERT-dependent)
3. **LLM=0.530 hard ceiling** — Gemini correctly penalizes:
   - "fluid accumulation" used for BOTH effusion (E2) AND pericardial effusion (E7) — same term, ambiguous
   - "fatty changes" extra FP entity (GT has 10 entities, pred has 11)
   - "collapse" vs "atelectasis" subtle clinical nuance
   - Degree imprecision

After applying C-2 and C-3: projected ~0.55–0.60.
But 5.3 (LLM=1.000 × FL_adj=0.613) = 0.613 stays above.

**Conclusion:** The system may be CORRECTLY identifying that 5.3's 7 clean precise pairs are higher quality than 5.0's 9 noisier pairs. The expected human ranking of 5.0 > 5.3 may need re-evaluation.

**Sample 5.1** (0.008): Correctly scored very low — genuinely wrong predictions. "atherosclerosis" ≠ "coronary artery calcification", "consolidation" ≠ "pericardial effusion", etc. Map correctly has no entries for these — they are different pathologies.

**Sample 5.4** (0.052): Correctly scored very low — "pleural thickening" ≠ "effusion", "hepatic steatosis" ≠ "hepatomegaly", wrong vessel for calcification, 3 presence contradictions.

## 17.4 Correlation with Human Rankings

| Dataset | Pearson r | Notes |
|---------|-----------|-------|
| GT4 | ~1.0 | Simple dataset, easy to rank |
| GT0 | ~0.99 | After Bug 3.1 fix |
| GT1 | ~0.99 | |
| GT2 | ~0.98 | |
| GT3 | ~0.97 | |
| GT6 | ~1.0 | Unit test — designed to be clear |
| GT5 | ~0.85 | 5.0 vs 5.3 order issue |

---

# 18. KNOWN SYSTEMATIC BIASES

## 18.1 LLM Ceiling Effect

When LLM evaluates a sample with high-quality TP pairs, it can score ~1.0 even if the sample has missed entities (FNs) or hallucinated ones (FPs). Coverage penalizes these via the FL_adj component, but the LLM component doesn't see FPs/FNs directly. This means:
- Sample with 5/10 GT entities but all 5 perfectly described: LLM~1.0, Coverage=0.5, Composite~0.5 ✓
- Sample with 10/10 GT entities but 5 hallucinations: LLM evaluates only the 5 correct TPs, LLM~1.0, Coverage=0.5, Composite~0.5 ✓

The system handles this correctly via Coverage. But it means LLM score alone is not a complete quality signal.

## 18.2 Duplicate Observation Term Penalty

When a pred uses the same observation term for two different GT entities (e.g., "fluid accumulation" for both effusion and pericardial effusion), the LLM evaluator sees this as a quality issue. The field scorer may still match correctly via compound matching, but the LLM penalizes the ambiguity. This is correct behavior — a good model should use distinct terms.

## 18.3 Extra Entity (FP) LLM Impact

If pred extracts an entity that is clinically IN the report but NOT in GT (e.g., "fatty changes" in sample5.0), this entity becomes a FP (reduces Coverage). The LLM evaluator, however, evaluates only TP pairs and doesn't directly see the FP. The LLM score may remain high while the Composite is pulled down by Coverage. This is the intended design — Coverage captures completeness, LLM captures quality.

## 18.4 Location Implicit in Observation (Gap A-1)

See Section 15. Multi-word observations that encode location (e.g., "bilateral pleural collection", "mediastinal lymph node enlargement", "pleural fluid") are penalized in location scoring when pred doesn't repeat the location word in the location array. This systematically biases against models that use descriptive observation strings.

## 18.5 SapBERT Threshold Sensitivity

The 0.70 threshold in location comparison and the 0.65 threshold in entity matching were set empirically. Changing these thresholds affects all rankings simultaneously. They should not be changed without re-validating GT0–GT4.

---

# 19. SESSION TRANSFER CHECKLIST

## 19.1 Files to Share with New Session (in order)

1. **This document** (`migration_blueprint_v2.md`) — primary context. Share FIRST.
2. **`V3_POST_FIX_AUDIT.txt`** — contains detailed per-sample results and validation data for GT0–GT4.
3. **`SUMMARY.txt` from GT5** — the current GT5 run showing 5.3 > 5.0 ordering issue.
4. **`.py` files on demand** — new session will ask. Have these ready:
   - `ulti_comp_mainn.py` (largest file, most changes needed)
   - `entity_level_evaluator.py` (Hungarian + FIX R-2)
   - `rule_based_validator.py` (field name references)
   - `llm_evaluator.py` (prompt template)
5. **`synonym_map.json`** — current 132-entry map. Share when synonym/compound work is being done.

## 19.2 Immediate Tasks for New Session

**Priority 1 — Apply pending code fixes to .py files:**
- C-1: Compound priority in `_compare_observation` Step 2
- C-2: Degree prefix stripping in `_compare_degree`
- C-3: Anatomical region fallback in `_compare_location_terms`
- A-1: Implicit-location exemption in `_compute_field_scores`

**Priority 2 — Validate GT5 ranking after fixes:**
- Expected: 5.2 > 5.0 > 5.3 > 5.4 > 5.1 (but see discussion — 5.0 vs 5.3 may be debatable)
- Critical check: Does 5.0 composite exceed 0.55 after C-2+C-3?
- Critical check: Does 5.3 composite stay below 5.0 after A-1?

**Priority 3 — GT7+ data expansion:**
- Apply to new GT datasets when available
- Check synonym_map coverage for new GT observation terms
- Add compound entries for any compound-split patterns not yet covered

## 19.3 What NOT to Bring to New Session

- `multi_embedding_evaluator.py` — deprecated, not imported anywhere
- `medical_schema_evaluator.py` — deprecated
- `comprehensive_evaluation.py` — deprecated
- Old schema (6-field) JSON files — do not confuse with new 4-field schema
- V2.0 evaluation code — everything was rebuilt for v3.0

## 19.4 Key Invariants (Do Not Change Without Full Re-Validation)

| Parameter | Current Value | Impact of Change |
|-----------|--------------|-----------------|
| Entity match threshold | 0.65 | Lower → more TPs (FPs get through); Higher → more FNs |
| Location token threshold | 0.70 | Affects loc scores across all samples |
| SapBERT obs thresholds | 0.85 / 0.70 | Affects field scorer quality |
| FIELD_WEIGHTS | obs=0.45, loc=0.35, deg=0.20 | Changes relative field importance |
| Coverage formula | TP/max(GT,Pred) | Using min would under-penalize FPs |
| Composite formula | FL_adj × LLM | Multiplicative zero-propagation is essential |

## 19.5 Quick Reference: Where Things Are

| Question | Answer |
|---------|--------|
| Where is the main entry point? | `ulti_comp_mainn.py` → `UserConfig` at top |
| Where is matching logic? | `entity_level_evaluator.py` → `match_entities()` |
| Where is field scoring? | `ulti_comp_mainn.py` → `_compute_field_scores()` |
| Where is obs comparison? | `ulti_comp_mainn.py` → `_compare_observation()` (4-step) |
| Where is the synonym map? | `synonym_map.json` (root dir) |
| Where are abbreviations? | `ulti_comp_mainn.py` → `LOCATION_ABBREVIATIONS`, `LOC_NORMMAP` |
| Where is presence gate? | `ulti_comp_mainn.py` → inside `_compute_field_scores()` |
| Where is contradiction detection? | `entity_level_evaluator.py` → `check_contradiction()` |
| Where are degree antonyms? | `ulti_comp_mainn.py` → `DEGREE_ANTONYMS` |
| Where is the composite formula? | `ulti_comp_mainn.py` → near end, after all TP scoring |

---

# APPENDIX A: COMPLETE _compare_observation CORRECT VERSION

The following is the CORRECT, BUG-FREE version of `_compare_observation` as it should exist after all fixes. Compare this against the actual .py file to find discrepancies:

```python
def _compare_observation(self,
                         gt_obs: str, pred_obs: str,
                         gt_locs: list = None,
                         pred_locs: list = None) -> Tuple[float, Optional[int]]:
    """
    Compare the 'observation' field.

    Returns: (score: float, consumed_pred_loc_idx: Optional[int])
    consumed_pred_loc_idx: index in pred_locs absorbed by compound match.
    Caller must remove this from pred_locs before location scoring.

    Step 1 — Exact string match → (1.0, None)
    Step 2 — Synonym map: resolve both to canonical.
             If match AND compound also fires → prefer compound (location consumption).
    Step 3A — DIR-A: gt_obs + gt_loc → compound key → compare to pred canonical.
    Step 3B — DIR-B: pred_obs + pred_loc → compound key → compare to gt canonical.
    Step 4 — SapBERT 3-zone fallback: >0.85→1.0, 0.70–0.85→partial, <0.70→0.0
    """
    gt_clean   = (gt_obs or '').lower().strip()
    pred_clean = (pred_obs or '').lower().strip()
    gt_locs    = gt_locs or []
    pred_locs  = pred_locs or []

    # ── Step 1 ────────────────────────────────────────────────────────────────
    if gt_clean == pred_clean:
        return 1.0, None
    if not gt_clean or not pred_clean:
        return 0.0, None

    # ── Step 2 ────────────────────────────────────────────────────────────────
    gt_canonical,   gt_conf   = _resolve_synonym(gt_clean)
    pred_canonical, pred_conf = _resolve_synonym(pred_clean)

    if gt_canonical == pred_canonical:
        # Synonym match — but check if compound also fires.
        # Prefer compound: it consumes a location token, preventing double-penalty.
        for ckey, idx in _build_compound_keys(pred_clean, pred_locs):
            entry = SYNONYM_MAP.get(ckey)
            if entry and entry['canonical'] == gt_canonical:
                return min(entry['confidence'], gt_conf), idx  # compound wins
        return min(gt_conf, pred_conf), None  # plain synonym

    # ── Step 3A: DIR-A ────────────────────────────────────────────────────────
    for compound_key, _loc_idx in _build_compound_keys(gt_clean, gt_locs):
        entry = SYNONYM_MAP.get(compound_key)
        if entry:
            cc, ccf = entry['canonical'], entry['confidence']
            if cc == pred_canonical:
                return min(ccf, pred_conf), None
            resolved, rc = _resolve_synonym(cc)
            if resolved == pred_canonical:
                return min(ccf, rc, pred_conf), None

    # ── Step 3B: DIR-B ────────────────────────────────────────────────────────
    for compound_key, loc_idx in _build_compound_keys(pred_clean, pred_locs):
        entry = SYNONYM_MAP.get(compound_key)
        if entry:
            cc, ccf = entry['canonical'], entry['confidence']
            if cc == gt_canonical:
                return min(ccf, gt_conf), loc_idx
            resolved, rc = _resolve_synonym(cc)
            if resolved == gt_canonical:
                return min(ccf, rc, gt_conf), loc_idx

    # ── Step 4 ────────────────────────────────────────────────────────────────
    sim = self._sapbert_similarity(gt_clean, pred_clean)  # NOT gt_term/pred_term
    if sim > 0.85:
        return 1.0, None
    elif sim > 0.70:
        return sim, None
    else:
        return 0.0, None

    # NOTE: _region_key / anatomical fallback does NOT belong here.
    # It belongs in _compare_location_terms (FIX C-3).
```

---

# APPENDIX B: COMPARISON TABLE FORMAT

The system outputs a comparison table in SUMMARY.txt:

```
COMPARISON TABLE
────────────────────────────────────────────────────────────────────
Sample     F1      FL(mean)  Coverage  FL(adj)   LLM     Composite  Rules
────────────────────────────────────────────────────────────────────
5.0        0.857   0.737     0.818     0.603     0.530   0.3196     0
5.1        0.476   0.220     0.455     0.100     0.080   0.0080     1
5.2        1.000   0.903     1.000     0.903     0.960   0.8669     0
5.3        0.700   0.875     0.700     0.613     1.000   0.6125     0
5.4        0.700   0.368     0.700     0.258     0.200   0.0515     1
```

Column meanings:
- **F1**: entity matching score (TP-based)
- **FL(mean)**: weighted field scores across TP pairs (before coverage)
- **Coverage**: TP / max(GT, Pred)
- **FL(adj)**: FL(mean) × Coverage
- **LLM**: Gemini score (0 if LLM disabled)
- **Composite**: FL(adj) × LLM — THE PRIMARY RANKING METRIC
- **Rules**: count of rule violations in pred

---

# APPENDIX C: DIRECTORY TREE (CURRENT)

```
project_root/
├── ulti_comp_mainn.py              ← MAIN ENTRY POINT
├── entity_level_evaluator.py       ← SapBERT + Hungarian matching
├── llm_evaluator.py                ← Gemini Pro evaluator
├── rule_based_validator.py         ← Schema consistency rules
├── modality_router.py              ← Report modality detection
├── synonym_map.json                ← 132-entry synonym/compound map
├── medical_schema_evaluator.py     ← DEPRECATED (ignore)
├── multi_embedding_evaluator.py    ← DEPRECATED (ignore)
├── migration_blueprint.md          ← OLD BLUEPRINT (superseded by this file)
├── V3_EXTENDED_AUDIT_GT1_GT4.txt
├── V3_POST_FIX_AUDIT.txt
├── workflows/
│   ├── base_workflow.py
│   └── chest_ct_workflow.py        ← Thoracic ontology
└── new_schema_v5/
    └── chest_ct/
        ├── 0/  ... 4/              ← GT0–GT4 data + results
        ├── 5/                      ← GT5 (current stress test)
        ├── 5_after_changesfor6/    ← GT5 run with GT6 code fixes
        ├── 5_beforechanges/        ← GT5 baseline (pre-fix)
        ├── 6/                      ← GT6 (unit test dataset)
        └── 6_beforechanges/        ← GT6 baseline
```

---

*End of Migration Blueprint v2. This document supersedes `migration_blueprint.md`.*
*Generated: 2026-02-27. Session covered: GT5 analysis, GT6 unit test, Actions 1-3, Fixes R-1/R-2, Bugs C-1/C-2/C-3/A-1.*
