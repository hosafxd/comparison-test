from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import json
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer

# Global model cache
MODEL_CACHE = {}

def load_model(name: str):
    if name not in MODEL_CACHE:
        print(f"🔄 Loading model: {name}")
        MODEL_CACHE[name] = SentenceTransformer(name)
    else:
        print(f"⚡ Using cached model: {name}")

    return MODEL_CACHE[name]


# ============================================================================
# YENİ EKLENECEK FONKSİYONLAR - BURAYA
# ============================================================================

def detect_entity_structural_errors(matches: List[Dict], gt_entities: List[Dict], pred_entities: List[Dict]) -> Dict:
    """
    Yapısal hataları tespit et:
    1. MERGED: 2+ GT entity -> 1 Pred entity 
    2. SPLIT: 1 GT entity -> 2+ Pred entity  
    3. DEGREE_MIXUP: Uyumsuz degree kombinasyonu
    """
    
    errors = {
        'merged_entities': [],      
        'split_entities': [],       
        'degree_mixups': [],        
        'contradictions': []        
    }
    
    # 1. MERGED ENTITY Detection
    # Her Pred entity için hangi GT entity'leriyle eşleştiğini bul
    pred_to_gt = defaultdict(list)
    
    for match in matches:
        if match['match_type'] == 'matched' and match['pred_entity']:
            # Pred entity'nin index'ini bul
            pred_idx = match.get('pred_idx')
            if pred_idx is None:
                # Eğer pred_idx yoksa, pred_entities listesinden bul
                try:
                    pred_idx = pred_entities.index(match['pred_entity'])
                except ValueError:
                    continue
            
            pred_to_gt[pred_idx].append({
                'gt_entity': match['gt_entity'],
                'gt_idx': matches.index(match)
            })
    
    # Birleştirilmiş entity'leri tespit et
    for pred_idx, gt_matches in pred_to_gt.items():
        if len(gt_matches) > 1:
            pred_ent = pred_entities[pred_idx]
            pred_finding = pred_ent.get('general_finding', 'Unknown')
            pred_degree = pred_ent.get('degree', [])
            
            # GT bilgilerini topla
            gt_findings = []
            for gm in gt_matches:
                gt_ent = gm['gt_entity']
                gt_findings.append({
                    'finding': gt_ent.get('general_finding'),
                    'specific': gt_ent.get('specific_finding'),
                    'degree': gt_ent.get('degree', []),
                    'location': gt_ent.get('location', [])
                })
            
            error_info = {
                'type': 'MERGED_ENTITY',
                'severity': 'HIGH',
                'description': f"{len(gt_matches)} farklı GT entity Sample'da tek entity'de birleştirilmiş",
                'gt_entities': gt_findings,
                'pred_entity': {
                    'finding': pred_finding,
                    'degree': pred_degree,
                    'location': pred_ent.get('location', [])
                },
                'impact': f"Recall düşüklüğü: {len(gt_matches)-1} entity kayıp (FN)"
            }
            errors['merged_entities'].append(error_info)
            
            # Degree mixup kontrolü
            if len(pred_degree) > 1 and _is_incompatible_degree_mix(pred_degree):
                errors['degree_mixups'].append({
                    'finding': pred_finding,
                    'degrees': pred_degree,
                    'source_entities': [g['finding'] for g in gt_findings],
                    'reason': "Farklı entity'lerden gelen degree'ler birleştirilmiş"
                })
    
    # 2. SPLIT ENTITY Detection (tersi durum)
    gt_to_pred = defaultdict(list)
    for match in matches:
        if match['match_type'] == 'matched' and match['gt_entity']:
            gt_idx = matches.index(match)
            gt_to_pred[gt_idx].append(match['pred_entity'])
    
    for gt_idx, pred_ents in gt_to_pred.items():
        if len(pred_ents) > 1:
            gt_ent = gt_entities[gt_idx]
            errors['split_entities'].append({
                'type': 'SPLIT_ENTITY',
                'severity': 'MEDIUM',
                'gt_entity': {
                    'finding': gt_ent.get('general_finding'),
                    'degree': gt_ent.get('degree')
                },
                'split_into': [p.get('general_finding') for p in pred_ents],
                'description': "1 GT entity Sample'da 2+ entity'ye bölünmüş"
            })
    
    # 3. CONTRADICTION Detection (presence çelişkileri)
    for match in matches:
        if match['match_type'] == 'matched':
            gt_ent = match['gt_entity']
            pred_ent = match['pred_entity']
            
            gt_pres = str(gt_ent.get('finding_presence', '')).lower()
            pred_pres = str(pred_ent.get('finding_presence', '')).lower()
            
            # Hard contradiction
            contradictions = [
                ('present', 'absent'),
                ('absent', 'present'),
                ('normal', 'abnormal'),
                ('enlarged', 'normal')
            ]
            
            if (gt_pres, pred_pres) in contradictions:
                errors['contradictions'].append({
                    'type': 'PRESENCE_CONTRADICTION',
                    'severity': 'CRITICAL',
                    'gt': gt_pres,
                    'pred': pred_pres,
                    'finding': gt_ent.get('general_finding'),
                    'description': f"GT: {gt_pres} vs Pred: {pred_pres} (tam çelişki)"
                })
    
    return errors


def _is_incompatible_degree_mix(degrees: List[str]) -> bool:
    """
    Uyumsuz degree kombinasyonu mu?
    Örn: ["mild", "trace"] -> biri Cardiomegaly'den, biri Effusion'dan geliyor olabilir
    """
    if not degrees or len(degrees) < 2:
        return False
    
    # Birbirini dışlayan dereceler (aynı anda olamazlar)
    exclusive_groups = [
        {'mild', 'moderate', 'severe'},           # Aynı anda farklı şiddet olamaz
        {'acute', 'chronic'},                      # Hem akut hem kronik olamaz
        {'trace', 'small', 'moderate', 'large'},   # Farklı miktarlar (eğer aynı finding değilse)
    ]
    
    degree_set = set(d.lower() for d in degrees if isinstance(d, str))
    
    for group in exclusive_groups:
        matches_in_group = degree_set & group
        if len(matches_in_group) > 1:
            return True
    
    return False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Semantic matching disabled.")


class SemanticMedicalMatcher:
    """
    Medical entity'ler için semantic matching (SapBERT tabanlı)
    """
    
    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings and EMBEDDING_AVAILABLE
        self.models = {}
        
        if self.use_embeddings:

            try:
                self.models['sapbert'] = load_model(
                    'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
                )
                print("✅ SapBERT loaded")
            except Exception as e:
                print(f"⚠️ SapBERT failed: {e}")
                self.models['sapbert'] = None

            try:
                self.models['pubmedbert'] = load_model(
                    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
                )
                print("✅ PubMedBERT loaded")
            except Exception as e:
                print(f"⚠️ PubMedBERT failed: {e}")
                self.models['pubmedbert'] = None

    
    def get_effective_finding(self, entity: Dict) -> Tuple[str, str]:
        """
        Hierarchical field handling: specific > general
        
        Returns:
            (finding_text, source_type)
        """
        specific = entity.get('specific_finding')
        general = entity.get('general_finding')
        
        # Normalize None/empty values
        def is_valid(val):
            if val is None:
                return False
            if isinstance(val, str):
                return val.lower() not in ['null', 'none', '', 'nan']
            return True
        
        if is_valid(specific):
            return str(specific), 'specific'
        elif is_valid(general):
            return str(general), 'general'
        else:
            return "", 'none'
    
    def check_contradiction(self, gt_entity: Dict, pred_entity: Dict) -> Dict:
        """
        Hard contradiction detection (çok kritik!)
        
        Returns:
            {'is_contradiction': bool, 'type': str, 'severity': str}
        """
        gt_presence = str(gt_entity.get('finding_presence', '')).lower().strip()
        pred_presence = str(pred_entity.get('finding_presence', '')).lower().strip()
        
        if not gt_presence or not pred_presence:
            return {'is_contradiction': False, 'type': 'none', 'severity': 'none'}
        
        # Hard contradictions (kesin çelişki)
        hard_contradictions = [
            ('present', 'absent'),
            ('absent', 'present'),
            ('normal', 'abnormal'),
            ('abnormal', 'normal'),
            ('enlarged', 'normal'),
            ('normal', 'enlarged'),
            ('thickening', 'normal'),
            ('present', 'normal'),  # Clinical context'e göre
        ]
        
        if (gt_presence, pred_presence) in hard_contradictions:
            return {
                'is_contradiction': True, 
                'type': 'hard',
                'severity': 'high',
                'reason': f"{gt_presence} vs {pred_presence}"
            }
        
        # Soft contradictions (belirsizlik)
        if 'uncertain' in [gt_presence, pred_presence] and gt_presence != pred_presence:
            return {
                'is_contradiction': False,  # Tam contradiction değil
                'type': 'uncertainty_mismatch',
                'severity': 'medium',
                'reason': f"certainty mismatch: {gt_presence} vs {pred_presence}"
            }
        
        return {'is_contradiction': False, 'type': 'none', 'severity': 'none'}
    
    def semantic_similarity(self, text1: str, text2: str, model_type: str = 'sapbert') -> float:
        """
        İki metin arasında semantic similarity (0-1)
        """
        if not self.use_embeddings or not self.models.get(model_type):
            # Fallback: String similarity
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        if not text1 or not text2:
            return 1.0 if text1 == text2 else 0.0
        
        model = self.models[model_type]
        embeddings = model.encode([text1, text2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(sim)
    
    def compare_location(self, loc1: List, loc2: List) -> float:
        """
        Location list'lerini karşılaştır (Semantic Jaccard)
        
        Örnek: ["pericardium"] vs ["heart", "pericardium"] → ~0.8
        """
        if not loc1 and not loc2:
            return 1.0
        
        if not loc1 or not loc2:
            return 0.0
        
        # Normalize
        def normalize_loc(loc):
            if isinstance(loc, list):
                return [str(l).lower().strip() for l in loc if l]
            return [str(loc).lower().strip()] if loc else []
        
        set1 = normalize_loc(loc1)
        set2 = normalize_loc(loc2)
        
        # Cross-similarity matrix (semantic)
        total_sim = 0
        matched = set()
        
        for l1 in set1:
            best_match = 0
            best_idx = -1
            
            for idx, l2 in enumerate(set2):
                if idx in matched:
                    continue
                
                # Exact match önce
                if l1 == l2:
                    sim = 1.0
                else:
                    # Semantic match (SapBERT)
                    sim = self.semantic_similarity(l1, l2, 'sapbert')
                
                if sim > best_match:
                    best_match = sim
                    best_idx = idx
            
            if best_match > 0.7:  # Threshold
                total_sim += best_match
                matched.add(best_idx)
        
        # F1-based scoring
        precision = total_sim / len(set1) if set1 else 0
        recall = total_sim / len(set2) if set2 else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_entity_similarity(self, gt_ent: Dict, pred_ent: Dict) -> Dict:
        """
        İki entity arasında comprehensive similarity hesapla
        
        Returns:
            {
                'overall_score': 0-1,
                'components': {...},
                'contradiction': {...},
                'is_match': bool
            }
        """
        # 1. Contradiction check (önce çelişki var mı bak)
        contradiction = self.check_contradiction(gt_ent, pred_ent)
        if contradiction['is_contradiction']:
            return {
                'overall_score': 0.0,
                'components': {},
                'contradiction': contradiction,
                'is_match': False
            }
        
        # 2. Finding comparison (Hierarchical)
        gt_finding, gt_type = self.get_effective_finding(gt_ent)
        pred_finding, pred_type = self.get_effective_finding(pred_ent)
        
        finding_sim = self.semantic_similarity(gt_finding, pred_finding, 'sapbert')
        
        # Type bonus: Eğer ikisi de specific ise ve match yaparsa +bonus
        type_bonus = 0.1 if (gt_type == 'specific' and pred_type == 'specific' and finding_sim > 0.8) else 0
        
        # 3. Location comparison
        gt_loc = gt_ent.get('location', [])
        pred_loc = pred_ent.get('location', [])
        location_sim = self.compare_location(gt_loc, pred_loc)
        
        # 4. Presence match (daha önce contradiction check yapıldı, burada sadece exact match)
        gt_pres = str(gt_ent.get('finding_presence', '')).lower()
        pred_pres = str(pred_ent.get('finding_presence', '')).lower()
        presence_sim = 1.0 if gt_pres == pred_pres else 0.5  # Contradiction yoksa partial
        
        # 5. Other fields (degree, measurement, etc.)
        other_sim = self._compare_other_fields(gt_ent, pred_ent)
        
        # Weighted combination
        weights = {
            'finding': 0.40,
            'location': 0.30,
            'presence': 0.20,
            'other': 0.10
        }
        
        overall = (
            weights['finding'] * finding_sim +
            weights['location'] * location_sim +
            weights['presence'] * presence_sim +
            weights['other'] * other_sim +
            type_bonus
        )
        
        # Clamp to 0-1
        overall = min(1.0, max(0.0, overall))
        
        return {
            'overall_score': overall,
            'components': {
                'finding_similarity': finding_sim,
                'finding_type': (gt_type, pred_type),
                'location_similarity': location_sim,
                'presence_similarity': presence_sim,
                'other_similarity': other_sim,
                'type_bonus': type_bonus
            },
            'contradiction': contradiction,
            'is_match': overall > 0.7  # Threshold
        }
    
    def _compare_other_fields(self, gt_ent: Dict, pred_ent: Dict) -> float:
        """Degree, measurement, comparison gibi alanları karşılaştır"""
        fields = ['degree', 'measurement', 'comparison']
        scores = []
        
        for field in fields:
            gt_val = gt_ent.get(field)
            pred_val = pred_ent.get(field)
            
            # Normalize empty
            def is_empty(val):
                if val is None:
                    return True
                if isinstance(val, str) and val.lower() in ['none', 'null', '']:
                    return True
                if isinstance(val, list) and len(val) == 0:
                    return True
                return False
            
            if is_empty(gt_val) and is_empty(pred_val):
                scores.append(1.0)
            elif is_empty(gt_val) or is_empty(pred_val):
                scores.append(0.0)
            else:
                # List comparison
                if isinstance(gt_val, list) and isinstance(pred_val, list):
                    gt_set = set(str(x).lower() for x in gt_val)
                    pred_set = set(str(x).lower() for x in pred_val)
                    intersection = len(gt_set & pred_set)
                    union = len(gt_set | pred_set)
                    scores.append(intersection / union if union > 0 else 0.0)
                else:
                    # String comparison with semantic similarity
                    sim = self.semantic_similarity(str(gt_val), str(pred_val), 'pubmedbert')
                    scores.append(sim)
        
        return np.mean(scores) if scores else 1.0


class EntityLevelEvaluator:
    """
    Main evaluator class - Entity Level Evaluation with Semantic Matching
    """
    
    def __init__(self, use_semantic_matching: bool = True, use_llm_for_borderline: bool = False, llm_evaluator=None):
        """
        Args:
            use_semantic_matching: SapBERT ile semantic matching kullan
            use_llm_for_borderline: Sadece kararsız durumlar için LLM kullan (maliyet düşürür)
            llm_evaluator: LLMEvaluator instance (opsiyonel)
        """
        self.semantic_matcher = SemanticMedicalMatcher(use_embeddings=use_semantic_matching)
        self.use_llm_for_borderline = use_llm_for_borderline
        self.llm_evaluator = llm_evaluator
        self.report_text = ""  # LLM için context

        self.field_weights = {
            "finding": 0.40,
            "location": 0.30,
            "presence": 0.20,
            "other": 0.10
        }
    
    def flatten_entities(self, schema: Dict) -> List[Dict]:
        """
        Schema'dan tüm entity'leri çıkar (input split'ten bağımsız)
        """
        all_entities = []
        
        for input_idx, input_item in enumerate(schema.get('inputs', [])):
            for entity in input_item.get('output', []):
                entity_with_meta = entity.copy()
                entity_with_meta['_source_input_idx'] = input_idx
                entity_with_meta['_source_input_text'] = input_item.get('input', '')[:100]
                all_entities.append(entity_with_meta)
        
        return all_entities
    
    def match_entities(self, gt_entities: List[Dict], pred_entities: List[Dict], report_text: str = "") -> List[Dict]:
        """
        GT ve Pred entity'leri arasında en iyi eşleştirmeyi bul
        (Hungarian algorithm yerine greedy matching - daha hızlı ve yeterli)
        """
        self.report_text = report_text
        
        matches = []
        used_pred_indices = set()
        
        for gt_idx, gt_entity in enumerate(gt_entities):
            best_match = None
            best_score = -1
            best_details = None
            
            for pred_idx, pred_entity in enumerate(pred_entities):
                if pred_idx in used_pred_indices:
                    continue
                
                # Semantic similarity hesapla
                sim_result = self.semantic_matcher.calculate_entity_similarity(gt_entity, pred_entity)
                score = sim_result['overall_score']
                
                # LLM fallback for borderline cases (0.3 < score < 0.7)
                if self.use_llm_for_borderline and self.llm_evaluator and 0.3 < score < 0.7:
                    llm_score = self._llm_entity_compare(gt_entity, pred_entity)
                    if llm_score is not None:
                        # Weighted average: 60% semantic, 40% LLM
                        score = 0.6 * score + 0.4 * llm_score
                        sim_result['llm_adjusted'] = True
                        sim_result['llm_score'] = llm_score
                
                if score > best_score:
                    best_score = score
                    best_match = pred_entity
                    best_details = sim_result
                    best_pred_idx = pred_idx
            
            # Eşleşme kalitesine göre karar ver
            if best_score > 0.5:  # Minimum threshold
                matches.append({
                    'gt_entity': gt_entity,
                    'pred_entity': best_match,
                    'match_score': best_score,
                    'match_type': 'matched',
                    'details': best_details,
                    'pred_idx': best_pred_idx
                })
                used_pred_indices.add(best_pred_idx)
            else:
                # False negative (GT'de var, pred'de yok)
                matches.append({
                    'gt_entity': gt_entity,
                    'pred_entity': None,
                    'match_score': 0.0,
                    'match_type': 'false_negative',
                    'details': None
                })
        
        # False positives (Pred'de var, GT'de yok)
        for pred_idx, pred_entity in enumerate(pred_entities):
            if pred_idx not in used_pred_indices:
                matches.append({
                    'gt_entity': None,
                    'pred_entity': pred_entity,
                    'match_score': 0.0,
                    'match_type': 'false_positive',
                    'details': None
                })
        
        return matches
    
    def _llm_entity_compare(self, gt_entity: Dict, pred_entity: Dict) -> Optional[float]:
        """LLM ile entity çiftini karşılaştır (borderline durumlar için)"""
        if not self.llm_evaluator:
            return None
        
        try:
            # LLM evaluator'u entity level'da çalışacak şekilde kullan
            # Eğer evaluate_entity_pair metodu yoksa, schema_pair kullanarak workaround
            gt_schema = {'input': '', 'output': [gt_entity]}
            pred_schema = {'input': '', 'output': [pred_entity]}
            
            result = self.llm_evaluator.evaluate_schema_pair(
                gt_schema, 
                pred_schema, 
                self.report_text[:500]  # Context
            )
            
            return result.get('similarity_score', 0.5)
        except Exception as e:
            print(f"    LLM comparison error: {e}")
            return None
    
    # entity_level_evaluator.py içindeki compute_metrics fonksiyonunu BU KODLA DEĞİŞTİRİN

    def compute_metrics(self, matches: List[Dict]) -> Dict:
        """
        Compute precision, recall, F1, and field-wise metrics
        """
        true_positives = sum(1 for m in matches if m['match_type'] == 'matched')
        false_positives = sum(1 for m in matches if m['match_type'] == 'false_positive')
        false_negatives = sum(1 for m in matches if m['match_type'] == 'false_negative')
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Field-wise accuracy
        # Field-wise accuracy from semantic components
        field_wise = defaultdict(list)

        for match in matches:

            if match['match_type'] != 'matched':
                continue

            details = match.get('details')
            if not isinstance(details, dict):
                continue

            comps = details.get('components')
            if not isinstance(comps, dict):
                continue

            # Map component names → metric fields
            mapping = {
                'finding_similarity': 'finding',
                'location_similarity': 'location',
                'presence_similarity': 'presence',
                'other_similarity': 'other'
            }

            for comp_name, field_name in mapping.items():
                score = comps.get(comp_name)
                if score is not None:
                    field_wise[field_name].append(score)

        
        field_wise_mean = {
            field: float(np.mean(scores)) if scores else 0.0
            for field, scores in field_wise.items()
        }

        
        # Weighted overall score
        overall_score = sum(
            field_wise_mean.get(field, 0.0) * weight
            for field, weight in self.field_weights.items()
        )
        

        
        # Match quality distribution stats
        matched_scores = [
            m['match_score']
            for m in matches
            if m['match_type'] == 'matched'
        ]

        avg_match_quality = np.mean(matched_scores) if matched_scores else 0.0

        score_variance = np.var(matched_scores) if matched_scores else 0.0
        score_std = np.std(matched_scores) if matched_scores else 0.0
        score_min = np.min(matched_scores) if matched_scores else 0.0
        score_max = np.max(matched_scores) if matched_scores else 0.0

       
        contradictions = []

        for m in matches:
            details = m.get('details')

            # details None mı kontrol et
            if details and isinstance(details, dict):

                contra = details.get('contradiction')

                if contra and isinstance(contra, dict):
                    if contra.get('is_contradiction'):
                        contradictions.append(m)

        
        return {
           
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,

            
            'score_variance': score_variance,
            'score_std': score_std,
            'score_min': score_min,
            'score_max': score_max,

            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'field_wise_accuracy': field_wise_mean,
            'overall_score': overall_score,
            'avg_match_quality': avg_match_quality,
            'contradiction_count': len(contradictions),
            'total_gt_entities': len([m for m in matches if m['gt_entity'] is not None]),
            'total_pred_entities': len([m for m in matches if m['pred_entity'] is not None])
        }
    def compute_metrics_with_error_analysis(self, matches: List[Dict], gt_entities: List[Dict], pred_entities: List[Dict]) -> Dict:
        """
        Standart metrikler + yapısal hata analizi
        """
        # 1. Standart metrikler
        base_metrics = self.compute_metrics(matches)
        
        # 2. Yapısal hata tespiti
        structural_errors = detect_entity_structural_errors(matches, gt_entities, pred_entities)
        
        # 3. Hata özetini ekle
        error_counts = {
            'merged_count': len(structural_errors['merged_entities']),
            'split_count': len(structural_errors['split_entities']),
            'degree_mixups': len(structural_errors['degree_mixups']),
            'contradictions': len(structural_errors['contradictions'])
        }
        
        return {
            **base_metrics,
            'structural_errors': structural_errors,
            'error_counts': error_counts,
            'has_structural_errors': any(v > 0 for v in error_counts.values()),
            'error_summary': self._format_error_summary(structural_errors)
        }
    
    def _format_error_summary(self, errors: Dict) -> str:
        """Hata özetini insan-readable formata çevir"""
        parts = []
        
        if errors['merged_entities']:
            parts.append(f"🔴 {len(errors['merged_entities'])} merged entity")
        if errors['split_entities']:
            parts.append(f"🟡 {len(errors['split_entities'])} split entity")
        if errors['degree_mixups']:
            parts.append(f"🟠 {len(errors['degree_mixups'])} degree mixup")
        if errors['contradictions']:
            parts.append(f"⚫ {len(errors['contradictions'])} contradiction")
        
        return " | ".join(parts) if parts else "✅ No structural errors"