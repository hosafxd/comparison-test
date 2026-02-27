# ============================================================================
# ENTITY-LEVEL EVALUATION v2.0 (Semantic Matching + Rule-Based Validation)
# ============================================================================
"""
Entity-level evaluation with SapBERT semantic matching
+ Structural Error Detection (Merged/Split entities)
+ Rule-Based Validation (Forte-style pipeline)
+ Modality-Specific Workflow (Chest CT ontology)
"""
import re
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime

# ─── V1.0 Imports (mevcut — değişmedi) ──────────────────────────────
from entity_level_evaluator import EntityLevelEvaluator, SemanticMedicalMatcher
from llm_evaluator import LLMEvaluator


# ─── V2.0 Imports (yeni eklenen) ────────────────────────────────────
from rule_based_validator import RuleBasedProcessor
from modality_router import ModalityRouter
from workflows.chest_ct_workflow import ChestCTWorkflow, ThoracicOntology


# ============================================================================
# ⭐ USER CONFIGURATION
# ============================================================================
class UserConfig:
    """
    ═══════════════════════════════════════════════════════════════════
    📝 EDIT THESE SETTINGS TO CONTROL WHICH MODELS TO USE
    ═══════════════════════════════════════════════════════════════════
    """
        # ----------------------------------------------------------------
    # 📁 DATA SETTINGS
    # ----------------------------------------------------------------
    DATA_DIR = "new_schema_v5/chest_ct/5"   # örnek path
    GT_FILENAME = "gt5.json"
    MATCH_THRESHOLD = 0.7
    # ----------------------------------------------------------------
    # 🔑 API KEYS
    # ----------------------------------------------------------------
    API_KEYS = {
        "gemini": "",
        #"gemma": "",           
        #"glm": "",             
        #"deepseek": "",        
    }
    
    # ----------------------------------------------------------------
    # 🤖 LLM MODELS TO USE
    # ----------------------------------------------------------------
    SELECTED_LLM_MODELS = [
        "gemini_pro",
    ]
    


LLM_MODELS = {
    "gemini_flash": {"type": "gemini", "name": "models/gemini-2.5-flash"},
    "gemini_pro": {"type": "gemini", "name": "models/gemini-2.5-pro"},
    #"gemma": {"type": "gemma", "name": "gemma-3-27b-it"},
    #"glm": {"type": "glm", "name": "glm-4-flash"},
    #"deepseek": {"type": "deepseek", "name": "deepseek-chat"}
}


# ============================================================================
# STARTUP DISPLAY
# ============================================================================

print("\n" + "="*70)
print("ENTITY-LEVEL EVALUATION v3.0 (Mixed-Method + Composite)")
print("="*70)

valid_llm_keys = set(LLM_MODELS.keys())
invalid_llm = [k for k in UserConfig.SELECTED_LLM_MODELS if k not in valid_llm_keys]
if invalid_llm:
    print(f"  WARNING: Invalid LLM models: {invalid_llm}")
    UserConfig.SELECTED_LLM_MODELS = [k for k in UserConfig.SELECTED_LLM_MODELS if k in valid_llm_keys]

print(f"\n  Data Directory: {UserConfig.DATA_DIR}")
print(f"  Ground Truth:   {UserConfig.GT_FILENAME}")
print(f"  Match Threshold: {UserConfig.MATCH_THRESHOLD}")
print(f"\n  LLM Models ({len(UserConfig.SELECTED_LLM_MODELS)}):")
for llm in UserConfig.SELECTED_LLM_MODELS:
    llm_info = LLM_MODELS[llm]
    has_key = "ok" if UserConfig.API_KEYS.get(llm_info['type']) else "NO KEY"
    print(f"    [{has_key}] {llm}: {llm_info['name']}")
print("="*70)


# ============================================================================
# HELPER: STRUCTURAL ERROR DETECTION
# ============================================================================

def detect_entity_structural_errors(
    matches: List[Dict], gt_entities: List[Dict], pred_entities: List[Dict]
) -> Dict:
    """
    Detect structural errors:
    1. MERGED: 2+ GT entity -> 1 Pred entity
    2. SPLIT: 1 GT entity -> 2+ Pred entity  
    3. DEGREE_MIXUP: incompatible degree combination
    4. CONTRADICTION: finding_presence conflict
    """
    errors = {
        'merged_entities': [],
        'split_entities': [],
        'degree_mixups': [],
        'contradictions': []
    }
    
    def get_pred_idx(pred_ent):
        try:
            return pred_entities.index(pred_ent)
        except ValueError:
            return None
    
    # 1. MERGED ENTITY Detection
    pred_to_gt = defaultdict(list)
    for match in matches:
        if match['match_type'] == 'matched' and match.get('pred_entity'):
            pred_idx = get_pred_idx(match['pred_entity'])
            if pred_idx is not None:
                pred_to_gt[pred_idx].append({
                    'gt_entity': match['gt_entity'],
                    'match_score': match.get('match_score', 0)
                })
    
    for pred_idx, gt_matches in pred_to_gt.items():
        if len(gt_matches) > 1:
            pred_ent = pred_entities[pred_idx]
            pred_finding = pred_ent.get('observation', 'Unknown')
            pred_degree  = pred_ent.get('degree', [])
            gt_findings = []
            for gm in gt_matches:
                gt_ent = gm['gt_entity']
                gt_findings.append({
                    'observation': gt_ent.get('observation'),
                    'degree': gt_ent.get('degree', []),
                    'location': gt_ent.get('location', [])
                })
            
            errors['merged_entities'].append({
                'type': 'MERGED_ENTITY',
                'severity': 'HIGH',
                'description': f"{len(gt_matches)} GT entities merged into one pred entity",
                'gt_entities': gt_findings,
                'pred_entity': {
                    'finding': pred_finding,
                    'degree': pred_degree,
                    'location': pred_ent.get('location', [])
                },
                'impact': f"{len(gt_matches)-1} entity(s) lost (FN)"
            })
            
            if len(pred_degree) > 1:
                degree_set = set(str(d).lower() for d in pred_degree)
                exclusive = {'mild', 'moderate', 'severe', 'trace', 'small', 'large'}
                if len(degree_set & exclusive) > 1:
                    errors['degree_mixups'].append({
                        'finding': pred_finding,
                        'degrees': pred_degree,
                        'source_entities': [g['finding'] for g in gt_findings],
                        'reason': "Degrees from different entities merged"
                    })
    
    # 2. SPLIT ENTITY Detection
    gt_to_pred = defaultdict(list)
    for match in matches:
        if match['match_type'] == 'matched' and match.get('gt_entity'):
            try:
                gt_idx = gt_entities.index(match['gt_entity'])
                gt_to_pred[gt_idx].append(match['pred_entity'])
            except ValueError:
                continue
    
    for gt_idx, pred_ents in gt_to_pred.items():
        if len(pred_ents) > 1:
            gt_ent = gt_entities[gt_idx]
            errors['split_entities'].append({
                'type': 'SPLIT_ENTITY',
                'severity': 'MEDIUM',
                'gt_entity': {
                    'observation': gt_ent.get('observation'),
                    'degree': gt_ent.get('degree')
                },
                'split_into': [p.get('observation') for p in pred_ents],
                'count': len(pred_ents)
            })
    
    # 3. CONTRADICTION Detection (also caught by entity_level_evaluator, but
    #    this catches any that leak through matching)
    for match in matches:
        if match['match_type'] == 'matched':
            gt_ent = match['gt_entity']
            pred_ent = match['pred_entity']
            gt_pres = str(gt_ent.get('observation_presence', '')).lower()
            pred_pres = str(pred_ent.get('observation_presence', '')).lower()
            
            hard_contradictions = [
                ('present', 'absent'), ('absent', 'present'),
                ('normal', 'abnormal'), ('abnormal', 'normal'),
                ('uncertain', 'present'), ('present', 'uncertain'),
                ('uncertain', 'absent'), ('absent', 'uncertain'),
            ]
            if (gt_pres, pred_pres) in hard_contradictions:
                errors['contradictions'].append({
                    'observation': gt_ent.get('observation'),
                    'gt_presence': gt_pres,
                    'pred_presence': pred_pres
                })
    
    return errors


# ============================================================================
# CLINICAL CONSTANTS (used by mixed-method field scorer)
# ============================================================================

# Degree antonyms — if GT has one side and Pred has the other → score 0.0
DEGREE_ANTONYMS = {
    frozenset({'patent', 'narrowed'}),
    frozenset({'patent', 'obstructed'}),
    frozenset({'patent', 'occluded'}),
    frozenset({'patent', 'stenosed'}),
    frozenset({'unlikely', 'suspicious'}),
    frozenset({'unlikely', 'probable'}),
    frozenset({'unlikely', 'likely'}),
    frozenset({'mild', 'severe'}),
    frozenset({'stable', 'worsening'}),
    frozenset({'stable', 'progressing'}),
    frozenset({'normal', 'abnormal'}),
    frozenset({'normal', 'enlarged'}),
    frozenset({'small', 'large'}),
    frozenset({'small', 'massive'}),
}

# Degree synonyms — normalize before comparison
DEGREE_SYNONYMS = {
    'upper limits of normal': 'borderline',
    'borderline enlarged': 'borderline',
    'physiologic': 'trace',
    'physiological': 'trace',
    'unremarkable': 'normal',
    'within normal limits': 'normal',
}
import os as _os
import json

# -------------------------------------------------
# Path resolution (script + notebook compatible)
# -------------------------------------------------
try:
    _BASE_DIR = _os.path.dirname(__file__)   # .py execution
except NameError:
    _BASE_DIR = _os.getcwd()                 # Notebook / interactive

_SYNONYM_MAP_PATH = _os.path.join(_BASE_DIR, "synonym_map.json")

# -------------------------------------------------
# Load synonym map
# -------------------------------------------------
try:
    with open(_SYNONYM_MAP_PATH, 'r') as _f:
        _SYNONYM_MAP_RAW = json.load(_f)

    # Strip comment keys, keep only real entries
    SYNONYM_MAP = {
        k: v for k, v in _SYNONYM_MAP_RAW.items()
        if not k.startswith('_')
        and isinstance(v, dict)
        and 'canonical' in v
    }

    print(f"  ✅ Synonym map loaded: {len(SYNONYM_MAP)} entries")

except FileNotFoundError:
    SYNONYM_MAP = {}
    print("  ⚠️  synonym_map.json not found — synonym lookup disabled")


def _resolve_synonym(text: str) -> tuple:
    """
    Look up text in synonym map.
    Returns (canonical_form, confidence) or (text_lowered, 1.0) if not found.
    """
    key = text.lower().strip()
    entry = SYNONYM_MAP.get(key)
    if entry:
        return entry['canonical'], entry['confidence']
    return key, 1.0


def _build_compound_keys(obs: str, locs: list) -> list:
    """
    Build compound lookup keys for obs+loc combinations.
    Returns list of (key_string, location_index_used) tuples.
    Format: "observation_text|location_token"
    """
    obs_clean = obs.lower().strip()
    results = []
    for i, loc in enumerate(locs):
        loc_clean = str(loc).lower().strip()
        # Both orders: "obs|loc" and "loc|obs"
        results.append((f"{obs_clean}|{loc_clean}", i))
        results.append((f"{loc_clean}|{obs_clean}", i))
        # Also try obs + full loc as phrase
        results.append((f"{obs_clean} {loc_clean}", i))
        results.append((f"{loc_clean} {obs_clean}", i))
    return results
# Common radiology abbreviations
LOCATION_ABBREVIATIONS = {
    'rij': 'right internal jugular',
    'lij': 'left internal jugular',
    'rsc': 'right subclavian',
    'lsc': 'left subclavian',
    'rsv': 'right subclavian vein',
    'lsv': 'left subclavian vein',
    'svc': 'superior vena cava',
    'ivc': 'inferior vena cava',
    'pa': 'pulmonary artery',
    'mpa': 'main pulmonary artery',
    'rpa': 'right pulmonary artery',
    'lpa': 'left pulmonary artery',
    'lad': 'left anterior descending artery',
    'rca': 'right coronary artery',
    'lcx': 'left circumflex artery',
    'rll': 'right lower lobe',
    'rul': 'right upper lobe',
    'rml': 'right middle lobe',
    'lll': 'left lower lobe',
    'lul': 'left upper lobe',
    'lmsb': 'left mainstem bronchus',
    'rmsb': 'right mainstem bronchus',
}

# Mixed-method field weights
FIELD_WEIGHTS = {
    'observation': 0.45,   # merged general+specific (was 0.25+0.20)
    'location':    0.35,   # raised from 0.30 (redistributed from removed measurement)
    'degree':      0.20,   # raised from 0.15 (same redistribution)
    # observation_presence: NOT in weights — kept as safety gate only (see UCM-3)
    # measurement: REMOVED — does not exist in new schema
}


# ============================================================================
# V1.0 BASE EVALUATOR CLASS
# ============================================================================

class EntityLevelReportEvaluator:
    
    def __init__(self, api_keys: dict, match_threshold: float = 0.65,
                 entity_matcher_model: str = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'):
        self.api_keys = api_keys
        self.match_threshold = match_threshold
        
        print("\n  Initializing Semantic Medical Matcher (SapBERT)...")
        self.semantic_matcher = SemanticMedicalMatcher(
            use_embeddings=True, model_name=entity_matcher_model
        )
        self.entity_evaluator = EntityLevelEvaluator(
            use_semantic_matching=True,
            use_llm_for_borderline=False,
            llm_evaluator=None
        )
        
        # Store reference to SapBERT model for mixed-method field scorer
        self._sapbert_model = getattr(self.semantic_matcher, 'model', None)
        
        print("  Evaluator ready!")
    
    # ──────────────────────────────────────────────────────────────────
    # MAIN EVALUATION
    # ──────────────────────────────────────────────────────────────────
    
    def evaluate_single(self, gt_path: str, pred_path: str,
                        llm_models: List[str] = None) -> Dict:
        
        with open(gt_path, 'r') as f:
            gt_schema = json.load(f)
        with open(pred_path, 'r') as f:
            pred_schema = json.load(f)
        
        result = {
            'gt_file': Path(gt_path).name,
            'pred_file': Path(pred_path).name,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        # Report check
        if gt_schema.get('report', '').strip() != pred_schema.get('report', '').strip():
            return {**result, 'status': 'error', 'error': 'Reports do not match'}
        
        # Entity extraction (deduplication happens inside flatten_entities)
        gt_entities = self.entity_evaluator.flatten_entities(gt_schema)
        pred_entities = self.entity_evaluator.flatten_entities(pred_schema)
        
        print(f"\n{'='*70}")
        print(f"  Evaluating: {Path(pred_path).name}")
        print(f"{'='*70}")
        print(f"  Entity Count: GT={len(gt_entities)} | Pred={len(pred_entities)}")
        
        # SapBERT Matching
        matches = self.entity_evaluator.match_entities(
            gt_entities, pred_entities, report_text=gt_schema.get('report', '')
        )
        
        # Structural Metrics
        base_metrics = self.entity_evaluator.compute_metrics(matches)
        
        # Structural error detection
        structural_errors = detect_entity_structural_errors(matches, gt_entities, pred_entities)
        
        # Per-field error analysis (exact match based)
        field_errors = self._analyze_field_errors(matches)
        
        result['entity_metrics'] = {
            'total_gt': len(gt_entities),
            'total_pred': len(pred_entities),
            'true_positives': base_metrics['true_positives'],
            'false_positives': base_metrics['false_positives'],
            'false_negatives': base_metrics['false_negatives'],
            'precision': base_metrics['precision'],
            'recall': base_metrics['recall'],
            'f1_score': base_metrics['f1_score'],
            'avg_match_quality': base_metrics['avg_match_quality'],
            'contradictions': base_metrics['contradiction_count'],
            'field_wise_accuracy': base_metrics.get('field_wise_accuracy', {}),
            'structural_errors': structural_errors,
            'field_error_analysis': field_errors
        }
        
        print(f"\n  STRUCTURAL (SapBERT Entity-Level):")
        print(f"    Precision: {base_metrics['precision']:.3f}")
        print(f"    Recall:    {base_metrics['recall']:.3f}")
        print(f"    F1-Score:  {base_metrics['f1_score']:.3f}")
        print(f"    TP: {base_metrics['true_positives']} | "
              f"FP: {base_metrics['false_positives']} | "
              f"FN: {base_metrics['false_negatives']}")
        
        if structural_errors['merged_entities']:
            print(f"    Merged Entities: {len(structural_errors['merged_entities'])}")
        if structural_errors['split_entities']:
            print(f"    Split Entities: {len(structural_errors['split_entities'])}")
        if structural_errors['contradictions']:
            print(f"    Contradictions: {len(structural_errors['contradictions'])}")
        
        # ==========================================
        # MIXED-METHOD FIELD-LEVEL SCORING
        # ==========================================
        # Each field uses the CORRECT comparison method:
        #   finding_presence  → EXACT MATCH (categorical gate)
        #   general_finding   → EXACT MATCH + SapBERT fallback at 0.85
        #   specific_finding  → SapBERT SIMILARITY (synonyms common here)
        #   location          → TERM-BY-TERM with abbreviation expansion
        #   degree            → EXACT SET + ANTONYM detection
        #   measurement       → NUMERIC distance with unit normalization
        
        tp_pairs = [(m['gt_entity'], m['pred_entity'])
                     for m in matches if m['match_type'] == 'matched']
        
        if tp_pairs:
            print(f"\n  FIELD-LEVEL QUALITY (Mixed-Method, {len(tp_pairs)} TP pairs):")
            
            all_pair_scores = []  # list of per-pair score dicts
            
            for gt_ent, pred_ent in tp_pairs:
                pair_scores = self._compute_field_scores(gt_ent, pred_ent)
                all_pair_scores.append(pair_scores)
            
            # Aggregate per-field means
            field_names = ['observation', 'location', 'degree']
            per_field_means = {}
            for fname in field_names:
                vals = [ps[fname] for ps in all_pair_scores if fname in ps]
                per_field_means[fname] = float(np.mean(vals)) if vals else 1.0
            
            # Weighted overall with PRESENCE as GATE
            pair_weighted_scores = []
            for (gt_ent, pred_ent), ps in zip(tp_pairs, all_pair_scores):
                # Safety gate: if presence somehow differs on a TP pair, zero the score.
                # In practice this never fires (SapBERT matching already blocks contradicting pairs),
                # but it provides a second layer of clinical correctness defense.
                gt_pres   = str(gt_ent.get('observation_presence', '')).lower()
                pred_pres = str(pred_ent.get('observation_presence', '')).lower()
                if gt_pres == pred_pres:
                    presence_gate = 1.0
                elif {gt_pres, pred_pres} & {'uncertain'}:
                    presence_gate = 0.3
                else:
                    presence_gate = 0.0   # present↔absent on a TP — should never happen
                
                field_score = sum(
                    FIELD_WEIGHTS[f] * ps.get(f, 1.0) for f in FIELD_WEIGHTS
                )
                gated_score = field_score * presence_gate
                pair_weighted_scores.append(gated_score)
            
            weighted_mean = float(np.mean(pair_weighted_scores))
            
            # Coverage: TP / max(GT, Pred) — penalizes missing/hallucinated entities
            total_gt = len(gt_entities)
            total_pred = len(pred_entities)
            tp_count = len(tp_pairs)
            coverage = tp_count / max(total_gt, total_pred) if max(total_gt, total_pred) > 0 else 0.0
            adjusted_score = weighted_mean * coverage
            
            result['field_level_scores'] = {
                'weighted_mean': round(weighted_mean, 4),
                'per_field': {k: round(v, 4) for k, v in per_field_means.items()},
                'coverage': round(coverage, 4),
                'adjusted_score': round(adjusted_score, 4),
                'tp_count': tp_count,
                'pair_scores': [round(s, 4) for s in pair_weighted_scores]
            }
            
            print(f"    Weighted Mean:  {weighted_mean:.3f}")
            print(f"    Coverage:       {coverage:.3f}  (TP={tp_count} / max(GT={total_gt}, Pred={total_pred}))")
            print(f"    Adjusted Score: {adjusted_score:.3f}  (mean × coverage)")
            
        else:
            result['field_level_scores'] = {
                'weighted_mean': 0.0,
                'per_field': {},
                'coverage': 0.0,
                'adjusted_score': 0.0,
                'tp_count': 0,
                'pair_scores': []
            }
            print(f"\n  FIELD-LEVEL: No TP pairs to score → 0.000")
        
        # ==========================================
        # LLM EVALUATION
        # ==========================================
        result['llm_scores'] = {}
        
        if llm_models:
            print(f"\n  LLM Evaluation:")
            
            for llm_key in llm_models:
                if llm_key not in LLM_MODELS:
                    continue
                
                llm_config = LLM_MODELS[llm_key]
                api_key = self.api_keys.get(llm_config['type'])
                
                if not api_key:
                    print(f"    No API key for {llm_key}")
                    continue
                
                print(f"    {llm_key}...", end=" ")
                
                try:
                    llm_eval = LLMEvaluator(
                        model_type=llm_config['type'],
                        model_name=llm_config['name'],
                        api_key=api_key
                    )
                    
                    llm_scores = []
                    matched_pairs = [m for m in matches if m['match_type'] == 'matched'][:5]
                    
                    for i, match in enumerate(matched_pairs):
                        try:
                            llm_result = llm_eval.evaluate_schema_pair(
                                {'output': [match['gt_entity']], 'input': ''},
                                {'output': [match['pred_entity']], 'input': ''},
                                gt_schema.get('report', '')[:500]
                            )
                            score = llm_result.get('similarity_score', 0)
                            if score is not None:
                                llm_scores.append(float(score))
                        except Exception as e:
                            print(f"\n      Pair {i+1} failed: {str(e)[:50]}")
                            continue
                    
                    if llm_scores:
                        avg_llm = float(np.mean(llm_scores))
                        result['llm_scores'][llm_key] = {
                            'mean': avg_llm,
                            'count': len(llm_scores)
                        }
                        print(f"ok {avg_llm:.3f} (n={len(llm_scores)})")
                    else:
                        print("no scores")
                        
                except Exception as e:
                    print(f"failed: {str(e)[:50]}")
        
        # ==========================================
        # COMPOSITE SCORE = FL_adjusted × LLM
        # ==========================================
        fl_adj = result.get('field_level_scores', {}).get('adjusted_score', 0.0)
        
        llm_mean = 0.0
        if result.get('llm_scores'):
            llm_vals = [v['mean'] for v in result['llm_scores'].values() if 'mean' in v]
            llm_mean = float(np.mean(llm_vals)) if llm_vals else 0.0
        
        llm_was_run = bool(result.get('llm_scores'))
        if llm_was_run:
            composite = fl_adj * llm_mean   # LLM=0 is a real score, not "missing"
        else:
            composite = fl_adj              # true fallback: no LLM available
        
        result['composite_score'] = round(composite, 4)
        
        print(f"\n  COMPOSITE SCORE: {composite:.4f}")
        print(f"    = FL_adjusted({fl_adj:.3f}) x LLM({llm_mean:.3f})")
        
        # Sample mismatches (for debugging)
        print(f"\n  Sample Mismatches:")
        fp_shown = fn_shown = 0
        for match in matches:
            if match['match_type'] == 'false_positive' and fp_shown < 2:
                ent = match['pred_entity']
                print(f"    FP: {ent.get('observation')} [{ent.get('observation_presence')}] @ {ent.get('location')}")
                fp_shown += 1
            if match['match_type'] == 'false_negative' and fn_shown < 2:
                ent = match['gt_entity']
                print(f"    FN: {ent.get('observation')} [{ent.get('observation_presence')}] @ {ent.get('location')}")
                fn_shown += 1
        # ─── NEW BLOCK (add after "Sample Mismatches" print block) ───────────────────
        # ==========================================
        # FP/FN ANALYSIS BRANCH
        # ==========================================
        # Runs independently of TP scoring. Does not affect composite score.
        # Purpose: characterize what was missed (FN) and what was hallucinated (FP).

        fp_entities = [m['pred_entity'] for m in matches if m['match_type'] == 'false_positive']
        fn_entities = [m['gt_entity']   for m in matches if m['match_type'] == 'false_negative']

        fp_analysis = _analyze_fp_fn_entities(fp_entities, 'FP')
        fn_analysis = _analyze_fp_fn_entities(fn_entities, 'FN')

        result['fp_fn_analysis'] = {
            'fp_count':    len(fp_entities),
            'fn_count':    len(fn_entities),
            # present_ratio = proportion of FP/FN entities that are "present" findings
            # High FP present_ratio → model hallucinating real findings (dangerous)
            # High FN present_ratio → model missing real findings (dangerous)
            'fp_severity': fp_analysis['present_ratio'],
            'fn_severity': fn_analysis['present_ratio'],
            'fp_detail':   fp_analysis,
            'fn_detail':   fn_analysis,
        }

        print(f"\n  FP/FN ANALYSIS:")
        print(f"    FP: {len(fp_entities)} hallucinated entities | "
            f"severity (% present): {fp_analysis['present_ratio']:.1%}")
        if fp_entities:
            print(f"    FP presence breakdown: "
                f"present={fp_analysis['presence_distribution'].get('present',0)} | "
                f"absent={fp_analysis['presence_distribution'].get('absent',0)} | "
                f"uncertain={fp_analysis['presence_distribution'].get('uncertain',0)}")

        print(f"    FN: {len(fn_entities)} missed GT entities | "
            f"severity (% present): {fn_analysis['present_ratio']:.1%}")
        if fn_entities:
            print(f"    FN presence breakdown: "
                f"present={fn_analysis['presence_distribution'].get('present',0)} | "
                f"absent={fn_analysis['presence_distribution'].get('absent',0)} | "
                f"uncertain={fn_analysis['presence_distribution'].get('uncertain',0)}")

        # Show up to 3 most clinically significant FNs (present GT entities that were missed)
        critical_fn = [e for e in fn_analysis['entities'] if e['observation_presence'] == 'present']
        if critical_fn:
            print(f"    Critical FN (missed 'present' GT findings, up to 3):")
            for e in critical_fn[:3]:
                print(f"      - {e['observation']} @ {e['location']}")

        # Show up to 3 most clinically significant FPs (present pred entities with no GT match)
        critical_fp = [e for e in fp_analysis['entities'] if e['observation_presence'] == 'present']
        if critical_fp:
            print(f"    Critical FP (hallucinated 'present' findings, up to 3):")
            for e in critical_fp[:3]:
                print(f"      + {e['observation']} @ {e['location']}")
        result['detailed_matches'] = matches
        return result
    
    # ──────────────────────────────────────────────────────────────────
    # MIXED-METHOD FIELD SCORER
    # ──────────────────────────────────────────────────────────────────
    
    def _compute_field_scores(self, gt_ent: Dict, pred_ent: Dict) -> Dict:

        scores = {}
        
         # ── LOCATION arrays (built first — needed by _compare_observation) ─────
        gt_loc   = gt_ent.get('location', []) or []
        pred_loc = pred_ent.get('location', []) or []
        if isinstance(gt_loc, str):   gt_loc   = [gt_loc]
        if isinstance(pred_loc, str): pred_loc = [pred_loc]
        gt_loc_clean   = [self._expand_abbreviation(str(l)).lower().strip()
                          for l in gt_loc if l and str(l) != 'None']
        pred_loc_clean = [self._expand_abbreviation(str(l)).lower().strip()
                          for l in pred_loc if l and str(l) != 'None']

        # ── OBSERVATION: synonym + compound + SapBERT ─────────────────────────
        # _compare_observation returns (score, consumed_pred_loc_idx).
        # consumed_pred_loc_idx: the index in pred_loc_clean that was absorbed
        # into the compound observation match (DIR-B). We strip it from the
        # location comparison to avoid double-penalizing the same token.
        obs_score, consumed_pred_loc_idx = self._compare_observation(
            gt_ent.get('observation') or '',
            pred_ent.get('observation') or '',
            gt_locs   = gt_loc_clean,
            pred_locs = pred_loc_clean
        )
        scores['observation'] = obs_score

        # ── LOCATION: strip consumed token before scoring ──────────────────────
        if consumed_pred_loc_idx is not None and consumed_pred_loc_idx < len(pred_loc_clean):
            pred_loc_for_scoring = [l for i, l in enumerate(pred_loc_clean)
                                    if i != consumed_pred_loc_idx]
        else:
            pred_loc_for_scoring = pred_loc_clean

        scores['location'] = self._compare_location_terms(gt_loc_clean, pred_loc_for_scoring)
        
        # ── DEGREE: exact set match + antonym detection (UNCHANGED) ────────────
        gt_deg   = [str(d).lower().strip() for d in (gt_ent.get('degree') or [])
                    if d and str(d) != 'None']
        pred_deg = [str(d).lower().strip() for d in (pred_ent.get('degree') or [])
                    if d and str(d) != 'None']
        if not gt_deg and not pred_deg:
            scores['degree'] = 1.0
        elif not gt_deg or not pred_deg:
            scores['degree'] = 0.0
        else:
            scores['degree'] = self._compare_degree(gt_deg, pred_deg)
        
        return scores
    
    # ──────────────────────────────────────────────────────────────────
    # FIELD COMPARISON HELPERS
    # ──────────────────────────────────────────────────────────────────
    
    def _sapbert_similarity(self, text1: str, text2: str) -> float:
        """Compute SapBERT cosine similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        if text1.lower().strip() == text2.lower().strip():
            return 1.0
        try:
            model = self._sapbert_model
            if model is None:
                return 1.0 if text1.lower() == text2.lower() else 0.0
            embeddings = model.encode([text1, text2])
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(max(0.0, sim))
        except Exception:
            return 1.0 if text1.lower() == text2.lower() else 0.0
    # ─── NEW METHOD (add after _sapbert_similarity at ~line 718) ──────────────────
    def _compare_observation(self, gt_obs: str, pred_obs: str,
                             gt_locs: list = None, pred_locs: list = None):
        """
        Compare the 'observation' field.
        Returns: Tuple[float, Optional[int]]
          - float: observation similarity score
          - Optional[int]: index into pred_locs that was CONSUMED by a compound
            match (DIR-B), or None if no compound fired. Used by _compute_field_scores
            to avoid double-penalizing that location token in the location scorer.

        Step 1 — Exact string match → 1.0
        Step 2 — Synonym map: resolve both obs to canonical, compare canonicals.
        Step 3 — Compound check:
                 DIR-A: gt_obs + gt_loc → compound → compare against pred canonical
                 DIR-B: pred_obs + pred_loc → compound → compare against gt canonical
                        RECORDS which pred_loc index was consumed.
        Step 4 — SapBERT 3-zone fallback.
        """
        gt_clean   = (gt_obs or '').lower().strip()
        pred_clean = (pred_obs or '').lower().strip()
        gt_locs    = gt_locs or []
        pred_locs  = pred_locs or []

        # ── Step 1: Exact match ───────────────────────────────────────────────
        if gt_clean == pred_clean:
            return 1.0, None
        if not gt_clean or not pred_clean:
            return 0.0, None

        # ── Step 2: Synonym map ───────────────────────────────────────────────
        gt_canonical,   gt_conf   = _resolve_synonym(gt_clean)
        pred_canonical, pred_conf = _resolve_synonym(pred_clean)

        if gt_canonical == pred_canonical:
            return min(gt_conf, pred_conf), None

        # ── Step 3A: Compound — DIR-A (GT obs + GT locs → compare to pred) ───
        for compound_key, _loc_idx in _build_compound_keys(gt_clean, gt_locs):
            entry = SYNONYM_MAP.get(compound_key)
            if entry:
                cc, ccf = entry['canonical'], entry['confidence']
                if cc == pred_canonical:
                    return min(ccf, pred_conf), None   # GT loc consumed — no pred penalty
                resolved, rc = _resolve_synonym(cc)
                if resolved == pred_canonical:
                    return min(ccf, rc, pred_conf), None

        # ── Step 3B: Compound — DIR-B (pred obs + pred locs → compare to GT) ─
        for compound_key, consumed_pred_idx in _build_compound_keys(pred_clean, pred_locs):
            entry = SYNONYM_MAP.get(compound_key)
            if entry:
                cc, ccf = entry['canonical'], entry['confidence']
                if cc == gt_canonical:
                    return min(ccf, gt_conf), consumed_pred_idx   # pred loc consumed!
                resolved, rc = _resolve_synonym(cc)
                if resolved == gt_canonical:
                    return min(ccf, rc, gt_conf), consumed_pred_idx

        # ── Step 4: SapBERT 3-zone fallback ──────────────────────────────────
        sim = self._sapbert_similarity(gt_clean, pred_clean)
        if sim > 0.85:
            return 1.0, None
        elif sim > 0.70:
            return sim, None
        else:
            return 0.0, None
    def _expand_abbreviation(self, text: str) -> str:
        """Expand common radiology abbreviations for better matching."""
        lower = text.lower().strip()
        return LOCATION_ABBREVIATIONS.get(lower, text)
    
    @staticmethod
    def _normalize_location_token(loc: str) -> str:
        """
        Normalize location token: expand abbreviations, strip trailing qualifiers.
        'pleural space' → 'pleural', 'left lung base' → 'left base',
        'osseous structures' → 'osseous', 'bilateral lungs' → 'bilateral'
        """
        loc = loc.lower().strip()
        # Trailing noise suffixes to strip
        suffixes = [' space', ' cavity', ' region', ' area', ' structures',
                    ' territory', ' field', ' zone', ' aspect']
        for s in suffixes:
            if loc.endswith(s):
                loc = loc[:-len(s)].strip()
        # "left lung base" → "left base"  /  "right lung base" → "right base"
        loc = loc.replace('left lung base', 'left base')
        loc = loc.replace('right lung base', 'right base')
        # "bilateral lungs" → "bilateral"
        if loc in ('bilateral lungs', 'both lungs'):
            loc = 'bilateral'
        return loc

    def _compare_location_terms(self, gt_terms: List[str], pred_terms: List[str]) -> float:
        """
        Term-by-term location comparison with SapBERT for fuzzy matching.
        Uses greedy best-match to handle reordering.
        Applies location token normalization before comparison.
        """
        if not gt_terms and not pred_terms:
            return 1.0
        if not gt_terms or not pred_terms:
            return 0.0

        # Normalize all tokens before comparison
        gt_terms   = [self._normalize_location_token(t) for t in gt_terms]
        pred_terms = [self._normalize_location_token(t) for t in pred_terms]

        # Remove empty strings that result from normalization
        gt_terms   = [t for t in gt_terms if t]
        pred_terms = [t for t in pred_terms if t]

        if not gt_terms and not pred_terms:
            return 1.0
        if not gt_terms or not pred_terms:
            return 0.0

        # Greedy match: for each GT term, find best matching pred term
        used_pred = set()
        matched_scores = []
        
        for gt_term in gt_terms:
            best_score = 0.0
            best_idx = -1
            for j, pred_term in enumerate(pred_terms):
                if j in used_pred:
                    continue
                if gt_term == pred_term:
                    score = 1.0
                else:
                    score = self._sapbert_similarity(gt_term, pred_term)
                if score > best_score:
                    best_score = score
                    best_idx = j
            
            if best_idx >= 0 and best_score > 0.7:
                used_pred.add(best_idx)
                matched_scores.append(best_score)
            else:
                matched_scores.append(0.0)  # unmatched GT term
        
        # Penalty for extra pred terms not in GT (hallucinated locations)
        extra_pred = len(pred_terms) - len(used_pred)
        total_terms = max(len(gt_terms), len(pred_terms))
        
        if total_terms == 0:
            return 1.0
        
        return sum(matched_scores) / total_terms
    
    @staticmethod
    def _normalize_degree_token(token: str) -> str:
        """Strip quantifier prefixes like 'up to', 'at least', 'approximately'."""
        token = token.lower().strip()
        prefixes = ['up to ', 'at least ', 'approximately ', 'about ', 'nearly ', 'almost ']
        for p in prefixes:
            if token.startswith(p):
                return token[len(p):].strip()
        return token

    def _compare_degree(self, gt_deg: List[str], pred_deg: List[str]) -> float:
        """Compare degree with antonym detection, synonym normalization, and prefix stripping."""
        # Normalize prefix tokens before comparison ("up to 1.2 cm" → "1.2 cm")
        gt_deg  = [self._normalize_degree_token(d) for d in gt_deg]
        pred_deg = [self._normalize_degree_token(d) for d in pred_deg]

        gt_set = set(gt_deg)
        pred_set = set(pred_deg)
        
        # Check for clinical antonyms first
        for pair in DEGREE_ANTONYMS:
            if (gt_set & pair) and (pred_set & pair) and (gt_set & pair) != (pred_set & pair):
                return 0.0  # OPPOSITE clinical meaning
        
        # Normalize synonyms
        gt_normalized = set(DEGREE_SYNONYMS.get(d, d) for d in gt_deg)
        pred_normalized = set(DEGREE_SYNONYMS.get(d, d) for d in pred_deg)
        
        # Jaccard similarity
        intersection = len(gt_normalized & pred_normalized)
        union = len(gt_normalized | pred_normalized)
        return intersection / union if union > 0 else 1.0
    

    
    # ──────────────────────────────────────────────────────────────────
    # PER-FIELD ERROR ANALYSIS (exact match based, for diagnostics)
    # ──────────────────────────────────────────────────────────────────
    
    def _analyze_field_errors(self, matches: List[Dict]) -> Dict:
        """Per-field error analysis: which fields cause most failures?"""
        field_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': []})
        
        for match in matches:
            if match['match_type'] != 'matched':
                continue
            
            gt_ent = match['gt_entity']
            pred_ent = match['pred_entity']
            
            fields = ['observation', 'observation_presence', 'location', 'degree']
            for field in fields:
                gt_val = gt_ent.get(field)
                pred_val = pred_ent.get(field) if pred_ent else None
                
                field_stats[field]['total'] += 1
                
                if self._values_match(gt_val, pred_val):
                    field_stats[field]['correct'] += 1
                else:
                    field_stats[field]['errors'].append({
                        'gt': str(gt_val)[:50],
                        'pred': str(pred_val)[:50]
                    })
        
        result = {}
        for field, stats in field_stats.items():
            if stats['total'] > 0:
                result[field] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'total': stats['total'],
                    'errors': len(stats['errors']),
                    'error_rate': (stats['total'] - stats['correct']) / stats['total']
                }
        return result
    
    def _values_match(self, val1, val2) -> bool:
        """Check if two values match (handles lists and None)."""
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False
        if isinstance(val1, list):
            val1 = set(str(x).lower() for x in val1 if x)
        else:
            val1 = str(val1).lower()
        if isinstance(val2, list):
            val2 = set(str(x).lower() for x in val2 if x)
        else:
            val2 = str(val2).lower()
        return val1 == val2


# ============================================================================
# V2.0 ENHANCED EVALUATOR CLASS (extends V1.0)
# ============================================================================

class RuleBasedEntityEvaluator(EntityLevelReportEvaluator):
    """
    Extends V1.0 with:
    Pipeline: Modality Detection -> Workflow Pre-process -> Rule-Based Validation
              -> V1.0 Evaluation -> Workflow Post-process
    """
    
    def __init__(self, api_keys: dict, match_threshold: float = 0.65):
        super().__init__(api_keys, match_threshold)
        self.modality_router = ModalityRouter()
        print("  V2.0 Rule-Based Validator + Modality Router initialized")
    
    def evaluate_single_v2(self, gt_path: str, pred_path: str,
                           llm_models: List[str] = None) -> Dict:
        """
        V2.0 evaluation pipeline:
        1. Load data
        2. Detect modality
        3. Workflow pre-process (entity normalization via ontology)
        4. Rule-based validation
        5. V1.0 evaluation (SapBERT matching + mixed-method FL + LLM + composite)
        6. Workflow post-process (category stats, conflict detection)
        """
        
        # 1. Load data
        with open(gt_path, 'r') as f:
            gt_schema = json.load(f)
        with open(pred_path, 'r') as f:
            pred_schema = json.load(f)
        
        # 2. MODALITY DETECTION
        gt_modality = self.modality_router.detect_modality(gt_schema.get('report', ''))
        pred_modality = self.modality_router.detect_modality(pred_schema.get('report', ''))
        print(f"\n  Detected Modality — GT: {gt_modality}, Pred: {pred_modality}")
        
        # 3. WORKFLOW PRE-PROCESS
        workflow = self.modality_router.get_workflow(pred_modality)
        workflow_stats = {}
        pred_entities_raw = self.entity_evaluator.flatten_entities(pred_schema)
        
        if workflow:
            pred_entities_enriched = workflow.pre_process(pred_entities_raw)
            print(f"    Workflow pre-processed {len(pred_entities_enriched)} entities")
        else:
            pred_entities_enriched = pred_entities_raw
        
        # 4. RULE-BASED VALIDATION
        rule_processor = RuleBasedProcessor(modality=pred_modality)
        processed_pred, violations = rule_processor.process(pred_entities_enriched)
        
        if violations:
            print(f"    Rule Violations: {len(violations)}")
            for v in violations[:5]:
                print(f"      [{v.severity.value}] {v.rule_id}: {v.message}")
            if len(violations) > 5:
                print(f"      ... and {len(violations) - 5} more")
        else:
            print(f"    No rule violations found")
        
        # 5. V1.0 EVALUATION (SapBERT matching + mixed-method FL + LLM + composite)
        result = super().evaluate_single(gt_path, pred_path, llm_models)
        
        # 6. WORKFLOW POST-PROCESS
        if workflow:
            _, workflow_stats = workflow.post_process(pred_entities_enriched, violations)
            print(f"    Ontology match rate: {workflow_stats.get('ontology_match_rate', 0):.1%}")
            if workflow_stats.get('conflicts_detected'):
                print(f"    Ontology conflicts: {len(workflow_stats['conflicts_detected'])}")
            if workflow_stats.get('unmatched_findings'):
                print(f"    Unmatched findings: {workflow_stats['unmatched_findings'][:3]}")
        
        # 7. ATTACH V2.0 METADATA
        result['v2_metadata'] = {
            'detected_modality': pred_modality,
            'rule_violations': [{
                'rule_id': v.rule_id,
                'severity': v.severity.value,
                'message': v.message,
                'field': v.field
            } for v in violations],
            'rule_violation_count': len(violations),
            'rule_violation_by_severity': {
                'error': sum(1 for v in violations if v.severity.value == 'error'),
                'warning': sum(1 for v in violations if v.severity.value == 'warning'),
                'info': sum(1 for v in violations if v.severity.value == 'info'),
            },
            'workflow_stats': workflow_stats,
            'validation_applied': True
        }
        
        return result


# ============================================================================
# STATISTICAL HELPERS
# ============================================================================

def bootstrap_confidence_interval(data: List[float], n_bootstrap: int = 1000,
                                   confidence: float = 0.95) -> Tuple[float, float]:
    if len(data) < 2:
        return (float(np.mean(data)), float(np.mean(data)))
    
    bootstrap_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))

# ─── NEW FUNCTION ─────────────────────────────────────────────────────────────
def _analyze_fp_fn_entities(entities: List[Dict], entity_type: str) -> Dict:
    """
    Characterize FP or FN entities by field distribution.

    For FP (hallucinated pred entities with no GT match):
      - presence_distribution tells us what the model hallucinated
      - present_ratio > 0.5 → dangerous: model asserts findings that don't exist

    For FN (missed GT entities that had no pred match):
      - presence_distribution tells us what the model missed
      - present_ratio > 0.5 → dangerous: model failed to find real findings

    location_fill_rate and degree_fill_rate indicate structural completeness
    of the unmatched entities (for FPs: are hallucinations at least well-formed?)
    """
    if not entities:
        return {
            'count': 0,
            'present_ratio': 0.0,
            'presence_distribution': {'present': 0, 'absent': 0, 'uncertain': 0},
            'location_fill_rate': 0.0,
            'degree_fill_rate': 0.0,
            'entities': []
        }

    presence_counts = {'present': 0, 'absent': 0, 'uncertain': 0, 'other': 0}
    loc_filled = 0
    deg_filled = 0

    for ent in entities:
        pres = str(ent.get('observation_presence', '')).lower().strip()
        if pres in ('present', 'absent', 'uncertain'):
            presence_counts[pres] += 1
        else:
            presence_counts['other'] += 1

        if ent.get('location'):
            loc_filled += 1
        if ent.get('degree'):
            deg_filled += 1

    n = len(entities)
    return {
        'count': n,
        'present_ratio': round(presence_counts['present'] / n, 3) if n > 0 else 0.0,
        'presence_distribution': {k: v for k, v in presence_counts.items() if k != 'other' or v > 0},
        'location_fill_rate': round(loc_filled / n, 3) if n > 0 else 0.0,
        'degree_fill_rate':   round(deg_filled / n, 3) if n > 0 else 0.0,
        'entities': [
            {
                'observation':          ent.get('observation', ''),
                'observation_presence': ent.get('observation_presence', ''),
                'location':             ent.get('location', []),
                'degree':               ent.get('degree', [])
            }
            for ent in entities
        ]
    }
# ============================================================================
# BATCH EVALUATION
# ============================================================================

def evaluate_directory_v2(
    data_dir: str = None,
    gt_filename: str = None,
    output_dir: str = None,
    llm_models: List[str] = None
):
    """Evaluate all samples in directory."""
    data_dir = data_dir or UserConfig.DATA_DIR
    gt_filename = gt_filename or UserConfig.GT_FILENAME
    output_dir = output_dir or (Path(data_dir) / 'entity_level_results')
    llm_models = llm_models or UserConfig.SELECTED_LLM_MODELS
    
    data_path = Path(data_dir)
    gt_path = data_path / gt_filename
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not gt_path.exists():
        print(f"  GT file not found: {gt_path}")
        return
    
    test_files = sorted(data_path.glob("sample*.json"))
    
    if not test_files:
        print(f"  No test files found in {data_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"BATCH EVALUATION")
    print(f"{'='*70}")
    print(f"  Directory:  {data_dir}")
    print(f"  GT:         {gt_filename}")
    print(f"  Test files: {len(test_files)}")
    
    # Initialize evaluator
    evaluator = RuleBasedEntityEvaluator(
        api_keys=UserConfig.API_KEYS,
        match_threshold=UserConfig.MATCH_THRESHOLD
    )
    
    # Evaluate each file
    all_results = []
    
    for test_file in test_files:
        result = evaluator.evaluate_single_v2(
            str(gt_path),
            str(test_file),
            llm_models=llm_models
        )
        all_results.append(result)
        
        # Save individual result
        result_file = output_path / f"result_{test_file.stem}.json"
        with open(result_file, 'w') as f:
            result_summary = {k: v for k, v in result.items() if k != 'detailed_matches'}
            json.dump(result_summary, f, indent=2, ensure_ascii=False)
        print(f"\n    Saved: {result_file.name}")
    
    # Generate summary
    _generate_summary(all_results, output_path)
    
    # ──────────────────────────────────────────────────────────────────
    # FINAL COMPARISON TABLE
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "="*100)
    print("  FINAL COMPARISON TABLE")
    print("="*100)
    
    header = (f"{'Sample':<12} {'F1':<8} {'FL(mean)':<10} {'Coverage':<10} "
          f"{'FL(adj)':<10} {'LLM':<8} {'Composite':<12} "
          f"{'FP_sev':<8} {'FN_sev':<8} {'Rules':<6}")
    print(header)
    print("-"*100)
    
    for r in all_results:
        if r.get('status') != 'success':
            print(f"{r['pred_file']:<12} ERROR")
            continue
        
        sample = r['pred_file'].replace('sample', '').replace('.json', '')
        f1 = r['entity_metrics']['f1_score']
        fl = r.get('field_level_scores', {})
        fl_mean = fl.get('weighted_mean', 0)
        fl_cov = fl.get('coverage', 0)
        fl_adj = fl.get('adjusted_score', 0)
        llm = r.get('llm_scores', {}).get('gemini_pro', {}).get('mean', 0)
        comp = r.get('composite_score', 0)
        rules = r.get('v2_metadata', {}).get('rule_violation_count', 0)
        
        fp_sev = r.get('fp_fn_analysis', {}).get('fp_severity', 0.0)
        fn_sev = r.get('fp_fn_analysis', {}).get('fn_severity', 0.0)
        print(f"{sample:<12} {f1:<8.3f} {fl_mean:<10.3f} {fl_cov:<10.3f} "
              f"{fl_adj:<10.3f} {llm:<8.3f} {comp:<12.4f} "
              f"{fp_sev:<8.2%} {fn_sev:<8.2%} {rules:<6d}")
    
    print("-"*100)
    print(f"\n  Column Guide:")
    print(f"    F1:        SapBERT entity matching (structural)")
    print(f"    FL(mean):  Mixed-method field quality on TP pairs (obs/loc/deg)")
    print(f"    Coverage:  TP / max(GT, Pred) — completeness penalty")
    print(f"    FL(adj):   FL(mean) x Coverage")
    print(f"    LLM:       Gemini clinical assessment of TP pairs")
    print(f"    Composite: FL(adj) x LLM — single best metric")
    print(f"    FP_sev:    % of hallucinated entities that are 'present' (higher = more dangerous)")
    print(f"    FN_sev:    % of missed GT entities that are 'present'  (higher = more dangerous)")
    print(f"    Rules:     Rule violations in pred schema (lower = better)")
    
    return all_results


# ============================================================================
# SUMMARY GENERATION
# ============================================================================

def _generate_summary(results: List[Dict], output_dir: Path):
    """Generate comprehensive summary report."""
    summary_path = output_dir / "SUMMARY.txt"
    
    successful = [r for r in results if r.get('status') == 'success']
    
    f1_scores = [r['entity_metrics']['f1_score'] for r in successful]
    f1_ci = bootstrap_confidence_interval(f1_scores) if f1_scores else (0, 0)
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ENTITY-LEVEL EVALUATION SUMMARY v3.0\n")
        f.write("(Mixed-Method Field Scoring + Composite)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total samples: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(results) - len(successful)}\n\n")
        
        # Confidence Intervals
        if f1_scores:
            f.write("CONFIDENCE INTERVALS (95% Bootstrap):\n")
            f.write("-"*70 + "\n")
            f.write(f"Structural F1: {np.mean(f1_scores):.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]\n\n")
        
        # Rule Violation Summary
        f.write("RULE-BASED VALIDATION SUMMARY:\n")
        f.write("-"*70 + "\n")
        for r in successful:
            v2 = r.get('v2_metadata', {})
            f.write(f"  {r['pred_file']}: {v2.get('rule_violation_count', 0)} violations "
                    f"(E:{v2.get('rule_violation_by_severity', {}).get('error', 0)} "
                    f"W:{v2.get('rule_violation_by_severity', {}).get('warning', 0)} "
                    f"I:{v2.get('rule_violation_by_severity', {}).get('info', 0)})")
            ws = v2.get('workflow_stats', {})
            if ws:
                f.write(f" | ontology: {ws.get('ontology_match_rate', 0):.0%}")
            f.write("\n")
        f.write("\n")
        
        # Per-field error analysis
        f.write("PER-FIELD ERROR ANALYSIS (exact match):\n")
        f.write("-"*70 + "\n")
        all_field_errors = defaultdict(lambda: {'total': 0, 'errors': 0})
        for r in successful:
            field_analysis = r['entity_metrics'].get('field_error_analysis', {})
            for field, stats in field_analysis.items():
                all_field_errors[field]['total'] += stats.get('total', 0)
                all_field_errors[field]['errors'] += stats.get('errors', 0)
        
        for field, stats in sorted(all_field_errors.items(), key=lambda x: x[1]['errors'], reverse=True):
            error_rate = stats['errors'] / stats['total'] * 100 if stats['total'] > 0 else 0
            f.write(f"  {field:25s}: {error_rate:5.1f}% error ({stats['errors']}/{stats['total']})\n")
        f.write("\n")
        
        # ──────────────────────────────────────────────────────────
        # MAIN COMPARISON TABLE
        # ──────────────────────────────────────────────────────────
        f.write("="*100 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Sample':<15} {'F1':<8} {'FL(mean)':<10} {'Coverage':<10} "
                f"{'FL(adj)':<10} {'LLM':<8} {'Composite':<12} {'Rules':<6}\n")
        f.write("-"*100 + "\n")
        
        for r in results:
            if r.get('status') != 'success':
                f.write(f"{r['pred_file']:<15} ERROR\n")
                continue
            
            sample = r['pred_file'].replace('sample', '').replace('.json', '')
            m = r['entity_metrics']
            fl = r.get('field_level_scores', {})
            llm = r.get('llm_scores', {}).get('gemini_pro', {}).get('mean', 0)
            comp = r.get('composite_score', 0)
            rules = r.get('v2_metadata', {}).get('rule_violation_count', 0)
            
            f.write(f"{sample:<15} {m['f1_score']:<8.3f} "
                    f"{fl.get('weighted_mean', 0):<10.3f} "
                    f"{fl.get('coverage', 0):<10.3f} "
                    f"{fl.get('adjusted_score', 0):<10.3f} "
                    f"{llm:<8.3f} {comp:<12.4f} {rules:<6d}\n")
        
        f.write("\n")
        
        # ──────────────────────────────────────────────────────────
        # PER-SAMPLE DETAILED RESULTS
        # ──────────────────────────────────────────────────────────
        f.write("="*70 + "\n")
        f.write("PER-SAMPLE DETAILED RESULTS\n")
        f.write("-"*70 + "\n")
        
        for result in results:
            if result.get('status') == 'error':
                f.write(f"\n{result['pred_file']}: ERROR - {result['error']}\n")
                continue
            
            metrics = result['entity_metrics']
            fl = result.get('field_level_scores', {})
            
            f.write(f"\n{result['pred_file']}:\n")
            
            # Structural
            f.write(f"  STRUCTURAL:\n")
            f.write(f"    Entities: GT={metrics['total_gt']}, Pred={metrics['total_pred']}\n")
            f.write(f"    Precision: {metrics['precision']:.3f}  "
                    f"Recall: {metrics['recall']:.3f}  "
                    f"F1: {metrics['f1_score']:.3f}\n")
            f.write(f"    TP={metrics['true_positives']}  "
                    f"FP={metrics['false_positives']}  "
                    f"FN={metrics['false_negatives']}\n")
            
            # Field-wise accuracy (SapBERT internal)
            if metrics.get('field_wise_accuracy'):
                f.write(f"  SAPBERT FIELD-WISE ACCURACY:\n")
                for field, acc in metrics['field_wise_accuracy'].items():
                    f.write(f"    {field}: {acc:.3f}\n")
            
            # Mixed-method field-level scores
            if fl.get('per_field'):
                f.write(f"  MIXED-METHOD FIELD SCORES (weighted mean: {fl.get('weighted_mean', 0):.3f}):\n")
                for fname, fscore in fl['per_field'].items():
                    f.write(f"    {fname:20s}: {fscore:.3f}\n")
                f.write(f"    Coverage:  {fl.get('coverage', 0):.3f}\n")
                f.write(f"    Adjusted:  {fl.get('adjusted_score', 0):.3f}\n")
            
            # LLM
            if result.get('llm_scores'):
                f.write(f"  LLM:\n")
                for llm_name, llm_data in result['llm_scores'].items():
                    f.write(f"    {llm_name}: {llm_data['mean']:.3f} (n={llm_data['count']})\n")
            
            # Composite
            f.write(f"  COMPOSITE: {result.get('composite_score', 0):.4f}\n")
            
            # Structural errors
            s_errors = metrics.get('structural_errors', {})
            if s_errors.get('merged_entities'):
                f.write(f"  MERGED ENTITIES: {len(s_errors['merged_entities'])}\n")
                for me in s_errors['merged_entities'][:2]:
                    f.write(f"    - {me['description']}\n")
            if s_errors.get('contradictions'):
                f.write(f"  CONTRADICTIONS: {len(s_errors['contradictions'])}\n")
                for co in s_errors['contradictions'][:3]:
                    f.write(f"    - {co.get('observation', co.get('finding', '?'))}: "
                            f"GT={co.get('gt_presence', '?')} vs Pred={co.get('pred_presence', '?')}\n")
        
        # ──────────────────────────────────────────────────────────
        # FIELD-LEVEL BREAKDOWN PER SAMPLE
        # ──────────────────────────────────────────────────────────
        f.write("\n" + "="*70 + "\n")
        f.write("FIELD-LEVEL SCORES PER SAMPLE\n")
        f.write("-"*70 + "\n")
        for r in results:
            if r.get('status') != 'success':
                continue
            fl = r.get('field_level_scores', {})
            per_field = fl.get('per_field', {})
            if per_field:
                f.write(f"\n  {r['pred_file']} (weighted: {fl.get('weighted_mean', 0):.3f}, "
                        f"coverage: {fl.get('coverage', 0):.3f}, "
                        f"adjusted: {fl.get('adjusted_score', 0):.3f}):\n")
                for fname, fscore in per_field.items():
                    f.write(f"    {fname:20s}: {fscore:.3f}\n")
        
        # ──────────────────────────────────────────────────────────
        # STRUCTURAL ERROR SUMMARY
        # ──────────────────────────────────────────────────────────
        f.write("\n" + "="*70 + "\n")
        f.write("STRUCTURAL ERROR ANALYSIS\n")
        f.write("-"*70 + "\n")
        
        all_merged = []
        all_splits = []
        all_contra = []
        
        for r in successful:
            errors = r['entity_metrics'].get('structural_errors', {})
            all_merged.extend(errors.get('merged_entities', []))
            all_splits.extend(errors.get('split_entities', []))
            all_contra.extend(errors.get('contradictions', []))
        
        f.write(f"Total Merged Entities: {len(all_merged)}\n")
        f.write(f"Total Split Entities:  {len(all_splits)}\n")
        f.write(f"Total Contradictions:  {len(all_contra)}\n")
        
        if all_merged:
            f.write(f"\nMERGED ENTITY EXAMPLES:\n")
            for i, me in enumerate(all_merged[:3], 1):
                f.write(f"  {i}. {me.get('description', 'N/A')}\n")
                f.write(f"     Severity: {me.get('severity', 'N/A')}\n")
                f.write(f"     Impact: {me.get('impact', 'N/A')}\n")
                gt_ents = me.get('gt_entities', [])
                if gt_ents:
                    findings = [g.get('finding', 'Unknown') for g in gt_ents]
                    f.write(f"     Source entities: {', '.join(findings)}\n")
                f.write("\n")
        
        if all_contra:
            f.write(f"\nCONTRADICTION EXAMPLES:\n")
            for i, co in enumerate(all_contra[:5], 1):
                # 'observation' is the correct key in the contradiction dict
                f.write(f"  {i}. {co.get('observation', co.get('finding', 'Unknown'))}: "
                        f"GT='{co.get('gt_presence', '?')}' vs "
                        f"Pred='{co.get('pred_presence', '?')}'\n")
        
        # ──────────────────────────────────────────────────────────
        # AGGREGATE STATISTICS
        # ──────────────────────────────────────────────────────────
        f.write("\n" + "="*70 + "\n")
        f.write("AGGREGATE STATISTICS\n")
        f.write("-"*70 + "\n")
        
        if successful:
            avg_precision = np.mean([r['entity_metrics']['precision'] for r in successful])
            avg_recall = np.mean([r['entity_metrics']['recall'] for r in successful])
            avg_f1 = np.mean([r['entity_metrics']['f1_score'] for r in successful])
            
            fl_adj_scores = [r.get('field_level_scores', {}).get('adjusted_score', 0) for r in successful]
            composite_scores = [r.get('composite_score', 0) for r in successful]
            
            f.write(f"\nAvg Structural Precision: {avg_precision:.3f}\n")
            f.write(f"Avg Structural Recall:    {avg_recall:.3f}\n")
            f.write(f"Avg Structural F1:        {avg_f1:.3f}\n")
            
            if fl_adj_scores:
                fl_ci = bootstrap_confidence_interval(fl_adj_scores)
                f.write(f"Avg FL(adjusted):         {np.mean(fl_adj_scores):.3f} "
                        f"[{fl_ci[0]:.3f}, {fl_ci[1]:.3f}]\n")
            
            if composite_scores:
                comp_ci = bootstrap_confidence_interval(composite_scores)
                f.write(f"Avg Composite:            {np.mean(composite_scores):.4f} "
                        f"[{comp_ci[0]:.4f}, {comp_ci[1]:.4f}]\n")
    
    print(f"\n  Summary saved: {summary_path}")


# ============================================================================
# RUN EVALUATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  STARTING EVALUATION")
    print("="*70)
    
    all_results = evaluate_directory_v2()
    
    print("\n  DONE!")
