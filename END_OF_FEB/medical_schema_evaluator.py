"""
Medical Schema Evaluation Pipeline
===================================
Multi-level evaluation system for comparing generated medical schemas 
against ground truth annotations.

STRUCTURAL EVALUATION ONLY
For embeddings: use multi_embedding_evaluator.py
For LLM: use llm_evaluator.py
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import hashlib

# Optional dependencies
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available - statistical tests disabled")


class MedicalSchemaEvaluator:

    FIELD_WEIGHTS_OLD = {
        'abnormality': 0.25,
        'finding': 0.02,
        'presence': 0.25,
        'location': 0.20,
        'degree': 0.15,
        'measurement': 0.10,
        'comparison': 0.03,
    }

    FIELD_WEIGHTS_NEW = {
        'general_finding': 0.20,
        'specific_finding': 0.20,
        'finding_presence': 0.30,
        'location': 0.15,
        'degree': 0.10,
        'measurement': 0.03,
        'comparison': 0.02
    }

    FIELD_WEIGHTS = FIELD_WEIGHTS_OLD

    def __init__(self):
        """Initialize structural evaluator"""
        pass


    def load_data(self, ground_truth_path: str, 
                predictions_path: str = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Load and preprocess data
        
        Returns:
            ground_truth, predictions (if available)
        """
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
            
        predictions = None
        if predictions_path:
            with open(predictions_path, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
                
        return ground_truth, predictions
    
    def detect_duplicates(self, data: List[Dict]) -> Dict[str, List[int]]:
        """
        Detect duplicate inputs for proper train/test splitting
        
        Returns:
            Dictionary mapping input hash to list of indices
        """
        input_map = defaultdict(list)
        
        for idx, item in enumerate(data):
            input_text = item.get('input', '').strip()
            input_hash = hashlib.md5(input_text.encode()).hexdigest()
            input_map[input_hash].append(idx)
            
        # Filter only duplicates
        duplicates = {k: v for k, v in input_map.items() if len(v) > 1}
        
        return duplicates
    
    def stratified_split(self, data: List[Dict], 
                        test_size: float = 0.2,
                        random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform stratified split avoiding duplicate leakage
        
        Strategy:
        1. Detect duplicates
        2. Group by input hash
        3. Split groups (not individual samples)
        4. Maintain class distribution
        """
        np.random.seed(random_state)
        
        # Group by input
        input_groups = defaultdict(list)
        for idx, item in enumerate(data):
            input_text = item.get('input', '').strip()
            input_hash = hashlib.md5(input_text.encode()).hexdigest()
            input_groups[input_hash].append(idx)
        
        # Get unique groups
        group_keys = list(input_groups.keys())
        np.random.shuffle(group_keys)
        
        # Split
        split_idx = int(len(group_keys) * (1 - test_size))
        train_groups = group_keys[:split_idx]
        test_groups = group_keys[split_idx:]
        
        # Reconstruct datasets
        train_indices = [idx for g in train_groups for idx in input_groups[g]]
        test_indices = [idx for g in test_groups for idx in input_groups[g]]
        
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        
        return train_data, test_data
    
    # ========================================================================
    # FORMAT DETECTION (NEW)
    # ========================================================================
    
    def _detect_schema_format(self, schema: Dict) -> str:
        """
        Detect if schema uses old or new field names
        
        Returns:
            'old', 'new', or 'unknown'
        """
        if 'output' in schema and schema['output']:
            first_entity = schema['output'][0]
            if 'general_finding' in first_entity:
                return 'new'
            elif 'abnormality' in first_entity:
                return 'old'
        return 'unknown'
    
    def _get_field_weights(self, schema: Dict) -> Dict:
        """
        Get appropriate field weights based on schema format
        
        Auto-detects format and returns corresponding weights
        """
        schema_format = self._detect_schema_format(schema)
        
        if schema_format == 'new':
            return self.FIELD_WEIGHTS_NEW
        else:
            return self.FIELD_WEIGHTS_OLD
    
    # ========================================================================
    # CORE COMPARISON LOGIC (UPDATED)
    # ========================================================================
    
    def compare_schemas(self, ground_truth: Dict, prediction: Dict) -> Dict[str, Any]:
        """
        Compare two schema outputs entity by entity
        
        AUTO-DETECTS schema format (old vs new) and uses appropriate weights
        
        Returns:
            Detailed comparison metrics including:
            - overall_score
            - field_scores
            - entity_matches
            - schema_format (detected)
        """
        gt_entities = ground_truth.get('output', [])
        pred_entities = prediction.get('output', [])
        
        # ⭐ AUTO-DETECT FORMAT AND GET WEIGHTS
        field_weights = self._get_field_weights(ground_truth)
        schema_format = self._detect_schema_format(ground_truth)
        
        results = {
            'num_gt_entities': len(gt_entities),
            'num_pred_entities': len(pred_entities),
            'entity_matches': [],
            'field_scores': defaultdict(list),
            'overall_score': 0.0,
            'schema_format': schema_format  # ← NEW: Track format
        }
        
        # Entity-level comparison
        for gt_entity in gt_entities:
            best_match = self._find_best_matching_entity(
                gt_entity, 
                pred_entities,
                field_weights  # ← Pass detected weights
            )
            results['entity_matches'].append(best_match)
     

            for field, weight in field_weights.items():
                score = self._compare_field(
                    gt_entity.get(field),
                    best_match['matched_entity'].get(field) if best_match['matched_entity'] else None
                )
                results['field_scores'][field].append(score)
                
                # ⭐ DEBUG: Print field comparisons
                if score < 1.0:
                    print(f"    Field '{field}': GT={gt_entity.get(field)} | Pred={best_match['matched_entity'].get(field) if best_match['matched_entity'] else None} | Score={score:.2f}")
                    
        # Calculate weighted overall score
        overall = 0.0
        for field, weight in field_weights.items():
            if results['field_scores'][field]:
                field_avg = np.mean(results['field_scores'][field])
                overall += field_avg * weight
                
        results['overall_score'] = overall
        
        return results
    
    def _find_best_matching_entity(self, 
                                   gt_entity: Dict, 
                                   pred_entities: List[Dict],
                                   field_weights: Dict) -> Dict:
        """
        Find best matching predicted entity for ground truth entity
        
        FORMAT-AWARE: Uses appropriate field names based on schema format
        
        Matching strategy:
        - Primary: location overlap (Jaccard similarity)
        - Secondary: abnormality/finding match
        """
        if not pred_entities:
            return {'matched_entity': None, 'similarity': 0.0}
        
        best_match = None
        best_score = -1.0
        
        # ⭐ FORMAT-AWARE FIELD NAMES
        if 'general_finding' in gt_entity:  # NEW format
            abnormality_key = 'general_finding'
        else:  # OLD format
            abnormality_key = 'abnormality'
        
        gt_location = set(gt_entity.get('location', []))
        gt_abnormality = str(gt_entity.get(abnormality_key, 'None')).lower()
    

        
  
        
        for pred_entity in pred_entities:
            pred_location = set(pred_entity.get('location', []))
            pred_abnormality = str(pred_entity.get(abnormality_key, 'None')).lower()
            
            # Location overlap (Jaccard similarity)
            if gt_location or pred_location:
                location_sim = len(gt_location & pred_location) / len(gt_location | pred_location)
            else:
                location_sim = 1.0 if gt_location == pred_location else 0.0
            
            # Abnormality match
            abnormality_match = 1.0 if gt_abnormality == pred_abnormality else 0.0
            
            # Combined score (weighted)
            score = 0.6 * location_sim + 0.4 * abnormality_match
            
            if score > best_score:
                best_score = score
                best_match = pred_entity
        
        return {'matched_entity': best_match, 'similarity': best_score}
    
    def _compare_field(self, gt_value: Any, pred_value: Any) -> float:
        """
        Compare individual field values
        
        Handles different types: strings, lists, None
        """
        # Both None or empty
        if self._is_empty(gt_value) and self._is_empty(pred_value):
            return 1.0
        
        # One empty, one not
        if self._is_empty(gt_value) or self._is_empty(pred_value):
            return 0.0
        
        # List comparison
        if isinstance(gt_value, list) and isinstance(pred_value, list):
            gt_set = set(str(x).lower() for x in gt_value)
            pred_set = set(str(x).lower() for x in pred_value)
            
            if not gt_set and not pred_set:
                return 1.0
            
            # Jaccard similarity
            intersection = len(gt_set & pred_set)
            union = len(gt_set | pred_set)
            return intersection / union if union > 0 else 0.0
        
        # String comparison
        gt_str = str(gt_value).lower().strip()
        pred_str = str(pred_value).lower().strip()
        
        # Exact match
        if gt_str == pred_str:
            return 1.0
        
        # Partial match (for measurements, etc.)
        if gt_str in pred_str or pred_str in gt_str:
            return 0.7
        
        return 0.0
    
    def _is_empty(self, value: Any) -> bool:
        """Check if value is effectively empty"""
        if value is None:
            return True
        if isinstance(value, str) and value.lower().strip() in ['none', '']:
            return True
        if isinstance(value, list) and (not value or all(self._is_empty(v) for v in value)):
            return True
        return False
    
    # ========================================================================
    # AGGREGATE METRICS
    # ========================================================================
    
    def calculate_aggregate_metrics(self, all_comparisons: List[Dict]) -> Dict:
        """
        Calculate aggregate metrics across all samples
        
        Returns:
            Dictionary with aggregate statistics
        """
        # Determine schema format from first comparison
        schema_format = all_comparisons[0].get('schema_format', 'old') if all_comparisons else 'old'
        
        # Use appropriate field weights
        if schema_format == 'new':
            field_weights = self.FIELD_WEIGHTS_NEW
        else:
            field_weights = self.FIELD_WEIGHTS_OLD
        
        # Field-wise averages
        field_wise_accuracy = {}
        for field in field_weights.keys():
            scores = []
            for comp in all_comparisons:
                if field in comp['field_scores']:
                    scores.extend(comp['field_scores'][field])
            field_wise_accuracy[field] = float(np.mean(scores)) if scores else 0.0
        
        # Overall metrics
        overall_scores = [comp['overall_score'] for comp in all_comparisons]
        
        metrics = {
            'schema_format': schema_format,
            'total_comparisons': len(all_comparisons),
            'exact_match_rate': float(np.mean([s == 1.0 for s in overall_scores])),
            'mean_score': float(np.mean(overall_scores)),
            'std_score': float(np.std(overall_scores)),
            'min_score': float(np.min(overall_scores)),
            'max_score': float(np.max(overall_scores)),
            'field_wise_accuracy': field_wise_accuracy,
            'field_weights_used': field_weights
        }
        
        return metrics
    
    def generate_evaluation_report(self, metrics: Dict, 
                                  output_path: str = 'evaluation_report.json'):
        """
        Generate comprehensive evaluation report
        """
        report = {
            'schema_format': metrics.get('schema_format', 'unknown'),
            'summary': {
                'total_comparisons': metrics['total_comparisons'],
                'exact_match_rate': f"{metrics['exact_match_rate']:.2%}",
                'mean_score': f"{metrics['mean_score']:.4f}",
                'std_score': f"{metrics['std_score']:.4f}",
                'score_range': f"[{metrics['min_score']:.4f}, {metrics['max_score']:.4f}]"
            },
            'field_wise_performance': {
                field: f"{score:.4f}" 
                for field, score in metrics['field_wise_accuracy'].items()
            },
            'field_weights_used': metrics['field_weights_used']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Evaluation report saved: {output_path}")
        
        return report


# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

class StatisticalAnalyzer:
    """
    Perform statistical tests to validate improvements
    
    Important for academic papers:
    - Bootstrap confidence intervals
    - Paired t-tests
    - Effect size calculation
    """
    
    @staticmethod
    def bootstrap_confidence_interval(scores: List[float], 
                                     n_bootstrap: int = 10000,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval
        
        Args:
            scores: List of evaluation scores
            n_bootstrap: Number of bootstrap iterations
            confidence: Confidence level (default: 0.95 for 95% CI)
        
        Returns:
            (lower_bound, upper_bound)
        """
        bootstrap_means = []
        n = len(scores)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    @staticmethod
    def compare_models(baseline_scores: List[float],
                      new_model_scores: List[float]) -> Dict:
        """
        Statistical comparison of two models
        
        Args:
            baseline_scores: Scores from baseline model
            new_model_scores: Scores from new model
        
        Returns:
            Dictionary with statistical test results:
            - p_value: Statistical significance
            - cohens_d: Effect size
            - significant: Whether difference is significant (p < 0.05)
            - improvement: Mean difference
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for statistical comparison. Install: pip install scipy")
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(new_model_scores, baseline_scores)
        
        # Effect size (Cohen's d)
        diff = np.array(new_model_scores) - np.array(baseline_scores)
        cohen_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohen_d),
            'significant': p_value < 0.05,
            'improvement': float(np.mean(new_model_scores) - np.mean(baseline_scores)),
            'baseline_mean': float(np.mean(baseline_scores)),
            'new_model_mean': float(np.mean(new_model_scores))
        }


# ============================================================================
# MODULE INFO
# ============================================================================

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        Medical Schema Evaluator - Structural Evaluation         ║
╚══════════════════════════════════════════════════════════════════╝

This module provides STRUCTURAL evaluation only.

For complete evaluation, also use:
- multi_embedding_evaluator.py (semantic similarity)
- llm_evaluator.py (LLM-based clinical validation)

Import this module in your evaluation scripts:
    from medical_schema_evaluator import MedicalSchemaEvaluator

Example:
    evaluator = MedicalSchemaEvaluator()
    result = evaluator.compare_schemas(ground_truth, prediction)
    print(f"Score: {result['overall_score']:.3f}")

For detailed usage, see comprehensive_evaluation.py
    """)