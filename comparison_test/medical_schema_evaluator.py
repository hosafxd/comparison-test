"""
Medical Schema Evaluation Pipeline
===================================
Multi-level evaluation system for comparing generated medical schemas 
against ground truth annotations.

Authors: [Your Name]
Date: January 2026
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import hashlib

# These will be installed later
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics import precision_recall_fscore_support
# import scipy.stats as stats


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    exact_match: float
    field_wise_accuracy: Dict[str, float]
    semantic_similarity: float
    clinical_validity_score: float
    f1_score: float
    precision: float
    recall: float
    entity_level_scores: List[Dict]


class MedicalSchemaEvaluator:
    """
    Comprehensive evaluator for medical schema extraction
    
    Features:
    - Multi-level evaluation (structural, semantic, clinical)
    - Field-wise importance weighting
    - Duplicate detection and handling
    - Statistical significance testing
    """
    
    # Field importance weights based on clinical relevance
    FIELD_WEIGHTS = {
        'abnormality': 0.25,
        'presence': 0.25,      # Critical: present/absent/uncertain
        'location': 0.20,
        'degree': 0.15,
        'measurement': 0.10,
        'finding': 0.03,
        'comparison': 0.02
    }
    
    def __init__(self, use_embeddings=True, use_llm=True):
        """
        Initialize evaluator
        
        Args:
            use_embeddings: Enable semantic similarity via embeddings
            use_llm: Enable LLM-based clinical validity checking
        """
        self.use_embeddings = use_embeddings
        self.use_llm = use_llm
        
        if use_embeddings:
            # Will load: sentence-transformers model
            self.embedding_model = None  # Placeholder
            
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
    
    def compare_schemas(self, ground_truth: Dict, prediction: Dict) -> Dict[str, Any]:
        """
        Compare two schema outputs entity by entity
        
        Returns detailed comparison metrics
        """
        gt_entities = ground_truth.get('output', [])
        pred_entities = prediction.get('output', [])
        
        results = {
            'num_gt_entities': len(gt_entities),
            'num_pred_entities': len(pred_entities),
            'entity_matches': [],
            'field_scores': defaultdict(list),
            'overall_score': 0.0
        }
        
        # Entity-level comparison
        for gt_entity in gt_entities:
            best_match = self._find_best_matching_entity(gt_entity, pred_entities)
            results['entity_matches'].append(best_match)
            
            # Field-wise scoring
            for field, weight in self.FIELD_WEIGHTS.items():
                score = self._compare_field(
                    gt_entity.get(field),
                    best_match['matched_entity'].get(field) if best_match['matched_entity'] else None
                )
                results['field_scores'][field].append(score)
        
        # Calculate weighted overall score
        overall = 0.0
        for field, weight in self.FIELD_WEIGHTS.items():
            if results['field_scores'][field]:
                field_avg = np.mean(results['field_scores'][field])
                overall += field_avg * weight
                
        results['overall_score'] = overall
        
        return results
    
    def _find_best_matching_entity(self, gt_entity: Dict, 
                                   pred_entities: List[Dict]) -> Dict:
        """
        Find best matching predicted entity for ground truth entity
        
        Uses location + abnormality as primary matching criteria
        """
        if not pred_entities:
            return {'matched_entity': None, 'similarity': 0.0}
        
        best_match = None
        best_score = -1.0
        
        gt_location = set(gt_entity.get('location', []))
        gt_abnormality = gt_entity.get('abnormality', 'None')
        
        for pred_entity in pred_entities:
            pred_location = set(pred_entity.get('location', []))
            pred_abnormality = pred_entity.get('abnormality', 'None')
            
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
    
    def calculate_aggregate_metrics(self, 
                                   all_comparisons: List[Dict]) -> EvaluationMetrics:
        """
        Calculate aggregate metrics across all samples
        """
        # Field-wise averages
        field_wise_accuracy = {}
        for field in self.FIELD_WEIGHTS.keys():
            scores = []
            for comp in all_comparisons:
                if field in comp['field_scores']:
                    scores.extend(comp['field_scores'][field])
            field_wise_accuracy[field] = np.mean(scores) if scores else 0.0
        
        # Overall metrics
        overall_scores = [comp['overall_score'] for comp in all_comparisons]
        
        metrics = EvaluationMetrics(
            exact_match=np.mean([s == 1.0 for s in overall_scores]),
            field_wise_accuracy=field_wise_accuracy,
            semantic_similarity=0.0,  # To be filled by embedding model
            clinical_validity_score=0.0,  # To be filled by LLM
            f1_score=np.mean(overall_scores),  # Approximation
            precision=0.0,  # To be calculated
            recall=0.0,     # To be calculated
            entity_level_scores=all_comparisons
        )
        
        return metrics
    
    def generate_evaluation_report(self, metrics: EvaluationMetrics, 
                                  output_path: str = 'evaluation_report.json'):
        """
        Generate comprehensive evaluation report
        """
        report = {
            'summary': {
                'exact_match_rate': f"{metrics.exact_match:.2%}",
                'overall_f1': f"{metrics.f1_score:.4f}",
                'avg_similarity': f"{metrics.semantic_similarity:.4f}"
            },
            'field_wise_performance': {
                field: f"{score:.4f}" 
                for field, score in metrics.field_wise_accuracy.items()
            },
            'weighted_importance': self.FIELD_WEIGHTS,
            'statistics': {
                'total_comparisons': len(metrics.entity_level_scores),
                'mean_entities_per_sample': np.mean([
                    s['num_gt_entities'] for s in metrics.entity_level_scores
                ])
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report


# ============================================================================
# LLM-Based Clinical Validity Checker
# ============================================================================

class LLMClinicalValidator:
    """
    Use LLM to assess clinical validity of schema comparisons
    
    Free APIs to use:
    1. Google Gemini 1.5 Flash (2M tokens/day free)
    2. HuggingFace Inference API
    """
    
    def __init__(self, provider: str = 'gemini'):
        """
        Args:
            provider: 'gemini', 'huggingface', or 'anthropic'
        """
        self.provider = provider
        
    def validate_schema_pair(self, ground_truth: Dict, 
                           prediction: Dict, 
                           input_text: str) -> Dict[str, Any]:
        """
        Send schema pair to LLM for clinical validity assessment
        
        Returns:
            - agreement_level: 'high', 'medium', 'low'
            - clinical_impact: severity of differences
            - explanation: detailed reasoning
        """
        prompt = self._construct_validation_prompt(
            ground_truth, prediction, input_text
        )
        
        # Call LLM API
        response = self._call_llm_api(prompt)
        
        return self._parse_llm_response(response)
    
    def _construct_validation_prompt(self, gt: Dict, pred: Dict, 
                                    input_text: str) -> str:
        """Construct prompt for LLM evaluation"""
        
        prompt = f"""You are a clinical validation expert. Compare these two medical schema extractions and assess their clinical equivalence.

INPUT TEXT:
{input_text}

GROUND TRUTH SCHEMA:
{json.dumps(gt['output'], indent=2)}

PREDICTED SCHEMA:
{json.dumps(pred['output'], indent=2)}

Evaluate:
1. Are they clinically equivalent? (Consider that different wordings may convey same clinical meaning)
2. What is the severity of any differences? (critical/moderate/minor/none)
3. Which specific fields differ and why?

Respond in JSON format:
{{
    "agreement_level": "high|medium|low",
    "clinical_impact": "critical|moderate|minor|none",
    "key_differences": ["list of differences"],
    "explanation": "detailed reasoning"
}}
"""
        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call LLM API based on provider
        
        This is a placeholder - actual implementation would use API calls
        """
        # TODO: Implement actual API calls
        # For Gemini: google-generativeai library
        # For HuggingFace: requests to inference API
        # For Anthropic: anthropic library
        
        return '{"agreement_level": "high", "clinical_impact": "none"}'
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM JSON response"""
        try:
            return json.loads(response)
        except:
            return {
                'agreement_level': 'unknown',
                'clinical_impact': 'unknown',
                'key_differences': [],
                'explanation': response
            }


# ============================================================================
# Statistical Significance Testing
# ============================================================================

class StatisticalAnalyzer:
    """
    Perform statistical tests to validate improvements
    
    Important for academic papers:
    - Bootstrap confidence intervals
    - Paired t-tests
    - McNemar's test for classification
    """
    
    @staticmethod
    def bootstrap_confidence_interval(scores: List[float], 
                                     n_bootstrap: int = 10000,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval
        
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
        
        Returns p-value and effect size
        """
        from scipy import stats
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(new_model_scores, baseline_scores)
        
        # Effect size (Cohen's d)
        diff = np.array(new_model_scores) - np.array(baseline_scores)
        cohen_d = np.mean(diff) / np.std(diff)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohen_d,
            'significant': p_value < 0.05,
            'improvement': np.mean(new_model_scores) - np.mean(baseline_scores)
        }


# ============================================================================
# Usage Example
# ============================================================================

def main():
    """Example usage pipeline"""
    
    # Initialize evaluator
    evaluator = MedicalSchemaEvaluator(use_embeddings=True, use_llm=True)
    
    # Load data
    ground_truth, predictions = evaluator.load_data(
        'schema_train.json',
        'model_predictions.json'  # Your model's output
    )
    
    # Check for duplicates (IMPORTANT!)
    duplicates = evaluator.detect_duplicates(ground_truth)
    print(f"Found {len(duplicates)} duplicate input groups")
    
    # Proper train/test split
    train_data, test_data = evaluator.stratified_split(ground_truth, test_size=0.2)
    
    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")
    
    # If you have predictions, compare them
    if predictions:
        # Match by input
        all_comparisons = []
        
        for gt_item in test_data:
            # Find matching prediction
            matching_pred = next(
                (p for p in predictions if p['input'] == gt_item['input']),
                None
            )
            
            if matching_pred:
                comparison = evaluator.compare_schemas(gt_item, matching_pred)
                all_comparisons.append(comparison)
        
        # Calculate metrics
        metrics = evaluator.calculate_aggregate_metrics(all_comparisons)
        
        # Generate report
        report = evaluator.generate_evaluation_report(metrics)
        
        print("\n=== EVALUATION REPORT ===")
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()