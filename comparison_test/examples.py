"""
Example Usage Scripts
====================
Practical examples of using the medical schema evaluation framework
"""

import json
import os
from evaluation_pipeline import ComprehensiveEvaluationPipeline
from medical_schema_evaluator import MedicalSchemaEvaluator, StatisticalAnalyzer
from llm_evaluator import LLMEvaluator, EmbeddingBasedEvaluator


# ============================================================================
# EXAMPLE 1: Basic Evaluation
# ============================================================================

def example_basic_evaluation():
    """
    Most common use case: evaluate your model against ground truth
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Evaluation")
    print("="*70)
    
    # Setup
    pipeline = ComprehensiveEvaluationPipeline(
        use_llm=True,
        use_embeddings=True,
        llm_provider='gemini'
    )
    
    # Run evaluation
    results = pipeline.run_complete_evaluation(
        ground_truth_path='/mnt/user-data/uploads/schema_train.json',
        predictions_path='your_model_predictions.json',
        output_dir='./results_basic',
        test_size=0.2,
        random_seed=42
    )
    
    # Print key metrics
    print(f"\n📊 RESULTS:")
    print(f"  F1 Score: {results['structural_evaluation']['overall_metrics']['f1_score']:.4f}")
    print(f"  Exact Match: {results['structural_evaluation']['overall_metrics']['exact_match_rate']:.2%}")
    
    for field, score in results['structural_evaluation']['field_wise_accuracy'].items():
        print(f"  {field:15s}: {score:.4f}")


# ============================================================================
# EXAMPLE 2: Comparing Two Models
# ============================================================================

def example_compare_models():
    """
    Compare performance of two different models
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparing Two Models")
    print("="*70)
    
    evaluator = MedicalSchemaEvaluator()
    
    # Load ground truth
    with open('/mnt/user-data/uploads/schema_train.json', 'r') as f:
        ground_truth = json.load(f)
    
    # Load predictions from both models
    with open('model_a_predictions.json', 'r') as f:
        predictions_a = json.load(f)
    
    with open('model_b_predictions.json', 'r') as f:
        predictions_b = json.load(f)
    
    # Split data
    train, test = evaluator.stratified_split(ground_truth, test_size=0.2)
    
    # Evaluate Model A
    scores_a = []
    for gt_item in test:
        pred = next((p for p in predictions_a if p['input'] == gt_item['input']), None)
        if pred:
            result = evaluator.compare_schemas(gt_item, pred)
            scores_a.append(result['overall_score'])
    
    # Evaluate Model B
    scores_b = []
    for gt_item in test:
        pred = next((p for p in predictions_b if p['input'] == gt_item['input']), None)
        if pred:
            result = evaluator.compare_schemas(gt_item, pred)
            scores_b.append(result['overall_score'])
    
    # Statistical comparison
    comparison = StatisticalAnalyzer.compare_models(scores_a, scores_b)
    
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"  Model A mean: {sum(scores_a)/len(scores_a):.4f}")
    print(f"  Model B mean: {sum(scores_b)/len(scores_b):.4f}")
    print(f"  Improvement: {comparison['improvement']:.4f}")
    print(f"  P-value: {comparison['p_value']:.4f}")
    print(f"  Significant: {'✓ YES' if comparison['significant'] else '✗ NO'}")
    print(f"  Effect size (Cohen's d): {comparison['cohens_d']:.4f}")


# ============================================================================
# EXAMPLE 3: Quick Duplicate Check
# ============================================================================

def example_check_duplicates():
    """
    Check your dataset for duplicates before training
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Duplicate Detection")
    print("="*70)
    
    evaluator = MedicalSchemaEvaluator()
    
    with open('/mnt/user-data/uploads/schema_train.json', 'r') as f:
        data = json.load(f)
    
    duplicates = evaluator.detect_duplicates(data)
    
    print(f"\n📋 DUPLICATE ANALYSIS:")
    print(f"  Total samples: {len(data)}")
    print(f"  Unique inputs: {len(data) - sum(len(v)-1 for v in duplicates.values())}")
    print(f"  Duplicate groups: {len(duplicates)}")
    print(f"  Total duplicates: {sum(len(v) for v in duplicates.values())}")
    
    # Show examples
    print(f"\n  Sample duplicate groups:")
    for idx, (hash_key, indices) in enumerate(list(duplicates.items())[:3]):
        print(f"    Group {idx+1}: {len(indices)} copies")
        print(f"      Indices: {indices}")
        print(f"      Input: {data[indices[0]]['input'][:60]}...")


# ============================================================================
# EXAMPLE 4: Field-Specific Analysis
# ============================================================================

def example_field_analysis():
    """
    Analyze performance on specific fields (e.g., 'presence' accuracy)
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Field-Specific Analysis")
    print("="*70)
    
    evaluator = MedicalSchemaEvaluator()
    
    with open('/mnt/user-data/uploads/schema_train.json', 'r') as f:
        ground_truth = json.load(f)
    
    with open('your_model_predictions.json', 'r') as f:
        predictions = json.load(f)
    
    # Focus on 'presence' field (critical for clinical decisions)
    presence_scores = []
    
    for gt in ground_truth[:50]:  # Sample for speed
        pred = next((p for p in predictions if p['input'] == gt['input']), None)
        
        if pred:
            gt_entities = gt.get('output', [])
            pred_entities = pred.get('output', [])
            
            for gt_entity in gt_entities:
                # Find matching entity in prediction
                best_match = evaluator._find_best_matching_entity(gt_entity, pred_entities)
                
                if best_match['matched_entity']:
                    # Compare presence field
                    score = evaluator._compare_field(
                        gt_entity.get('presence'),
                        best_match['matched_entity'].get('presence')
                    )
                    presence_scores.append({
                        'gt': gt_entity.get('presence'),
                        'pred': best_match['matched_entity'].get('presence'),
                        'score': score,
                        'input': gt['input'][:50]
                    })
    
    # Analyze results
    import numpy as np
    
    perfect = sum(1 for s in presence_scores if s['score'] == 1.0)
    errors = [s for s in presence_scores if s['score'] < 1.0]
    
    print(f"\n📊 PRESENCE FIELD ANALYSIS:")
    print(f"  Total entities: {len(presence_scores)}")
    print(f"  Perfect matches: {perfect} ({perfect/len(presence_scores)*100:.1f}%)")
    print(f"  Errors: {len(errors)}")
    print(f"  Mean score: {np.mean([s['score'] for s in presence_scores]):.4f}")
    
    # Show error examples
    print(f"\n  Error examples:")
    for error in errors[:5]:
        print(f"    GT: {error['gt']:10s} | Pred: {error['pred']:10s} | Text: {error['input']}...")


# ============================================================================
# EXAMPLE 5: LLM-Only Quick Evaluation
# ============================================================================

def example_llm_quick_eval():
    """
    Use only LLM for quick qualitative assessment
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: LLM-Only Quick Evaluation")
    print("="*70)
    
    # Initialize LLM evaluator
    llm_eval = LLMEvaluator(provider='gemini')
    
    # Sample pair
    ground_truth = {
        'input': 'Spleen size is enlarged and measured 134 mm.',
        'output': [
            {
                'abnormality': 'enlarged',
                'presence': 'present',
                'location': ['spleen'],
                'measurement': '134 mm'
            }
        ]
    }
    
    prediction = {
        'input': 'Spleen size is enlarged and measured 134 mm.',
        'output': [
            {
                'abnormality': 'splenomegaly',  # Different term, same meaning
                'presence': 'present',
                'location': ['spleen'],
                'measurement': '134mm'  # No space
            }
        ]
    }
    
    # Evaluate
    result = llm_eval.evaluate_schema_pair(
        ground_truth,
        prediction,
        ground_truth['input']
    )
    
    print(f"\n📊 LLM EVALUATION RESULT:")
    print(f"  Similarity Score: {result.get('similarity_score', 0):.4f}")
    print(f"  Clinical Equivalence: {result.get('clinical_equivalence', 'unknown')}")
    print(f"  Same Meaning: {result.get('are_same_meaning', False)}")
    print(f"\n  Assessment: {result.get('overall_assessment', 'N/A')}")


# ============================================================================
# EXAMPLE 6: Semantic Similarity Only
# ============================================================================

def example_semantic_similarity():
    """
    Compare schemas using only embeddings (fast, no API needed)
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Semantic Similarity (Embeddings)")
    print("="*70)
    
    try:
        emb_eval = EmbeddingBasedEvaluator()
        
        schema1 = {
            'output': [{
                'abnormality': 'fracture',
                'presence': 'absent',
                'location': ['skull']
            }]
        }
        
        schema2 = {
            'output': [{
                'abnormality': 'bone break',  # Synonym
                'presence': 'not present',    # Paraphrase
                'location': ['cranium']       # Medical term
            }]
        }
        
        similarity = emb_eval.compute_schema_similarity(schema1, schema2)
        
        print(f"\n📊 SEMANTIC SIMILARITY:")
        print(f"  Score: {similarity:.4f}")
        print(f"  Interpretation: {'High' if similarity > 0.8 else 'Medium' if similarity > 0.6 else 'Low'}")
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Make sure sentence-transformers is installed:")
        print("  pip install sentence-transformers")


# ============================================================================
# EXAMPLE 7: Error Analysis Report
# ============================================================================

def example_error_analysis():
    """
    Generate detailed error analysis report
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Detailed Error Analysis")
    print("="*70)
    
    evaluator = MedicalSchemaEvaluator()
    
    with open('/mnt/user-data/uploads/schema_train.json', 'r') as f:
        ground_truth = json.load(f)
    
    with open('your_model_predictions.json', 'r') as f:
        predictions = json.load(f)
    
    # Evaluate all
    all_results = []
    for gt in ground_truth[:100]:  # Sample
        pred = next((p for p in predictions if p['input'] == gt['input']), None)
        if pred:
            result = evaluator.compare_schemas(gt, pred)
            result['input'] = gt['input']
            all_results.append(result)
    
    # Categorize errors
    low_score = [r for r in all_results if r['overall_score'] < 0.5]
    medium_score = [r for r in all_results if 0.5 <= r['overall_score'] < 0.8]
    high_score = [r for r in all_results if r['overall_score'] >= 0.8]
    
    print(f"\n📊 ERROR DISTRIBUTION:")
    print(f"  Low score (<0.5): {len(low_score)} samples")
    print(f"  Medium score (0.5-0.8): {len(medium_score)} samples")
    print(f"  High score (≥0.8): {len(high_score)} samples")
    
    # Analyze low-score cases
    print(f"\n  Common issues in low-score samples:")
    
    entity_mismatch = sum(1 for r in low_score if r['num_gt_entities'] != r['num_pred_entities'])
    print(f"    Entity count mismatch: {entity_mismatch}/{len(low_score)}")
    
    # Field-specific issues
    for field in ['abnormality', 'presence', 'location']:
        low_field_scores = [
            r for r in low_score 
            if field in r['field_scores'] and 
            sum(r['field_scores'][field]) / len(r['field_scores'][field]) < 0.5
        ]
        print(f"    {field} errors: {len(low_field_scores)}/{len(low_score)}")


# ============================================================================
# EXAMPLE 8: Cross-Validation Setup
# ============================================================================

def example_cross_validation():
    """
    Setup for k-fold cross-validation
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: K-Fold Cross-Validation Setup")
    print("="*70)
    
    evaluator = MedicalSchemaEvaluator()
    
    with open('/mnt/user-data/uploads/schema_train.json', 'r') as f:
        data = json.load(f)
    
    # Detect duplicates first
    duplicates = evaluator.detect_duplicates(data)
    
    # Group by unique inputs
    from collections import defaultdict
    import hashlib
    import numpy as np
    
    input_groups = defaultdict(list)
    for idx, item in enumerate(data):
        input_hash = hashlib.md5(item['input'].encode()).hexdigest()
        input_groups[input_hash].append(idx)
    
    unique_groups = list(input_groups.keys())
    
    # K-fold split
    k = 5
    fold_size = len(unique_groups) // k
    
    print(f"\n📊 CROSS-VALIDATION SETUP:")
    print(f"  Total samples: {len(data)}")
    print(f"  Unique inputs: {len(unique_groups)}")
    print(f"  K-folds: {k}")
    print(f"  Samples per fold: ~{fold_size * sum(len(v) for v in input_groups.values()) // len(unique_groups)}")
    
    # Create folds
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k-1 else len(unique_groups)
        
        test_groups = unique_groups[start:end]
        train_groups = [g for g in unique_groups if g not in test_groups]
        
        test_indices = [idx for g in test_groups for idx in input_groups[g]]
        train_indices = [idx for g in train_groups for idx in input_groups[g]]
        
        folds.append({
            'fold': i + 1,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'train_size': len(train_indices),
            'test_size': len(test_indices)
        })
        
        print(f"  Fold {i+1}: Train={len(train_indices)}, Test={len(test_indices)}")
    
    # Save folds for reproducibility
    with open('cv_folds.json', 'w') as f:
        json.dump(folds, f, indent=2)
    
    print(f"\n  ✓ Folds saved to cv_folds.json")


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """Run all examples or select one"""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          Medical Schema Evaluation - Example Scripts            ║
╚══════════════════════════════════════════════════════════════════╝

Select an example to run:

1. Basic Evaluation
2. Compare Two Models
3. Check for Duplicates
4. Field-Specific Analysis
5. LLM Quick Evaluation
6. Semantic Similarity
7. Error Analysis Report
8. Cross-Validation Setup

0. Run all examples

Enter your choice (0-8): """)
    
    try:
        choice = input().strip()
        
        examples = {
            '1': example_basic_evaluation,
            '2': example_compare_models,
            '3': example_check_duplicates,
            '4': example_field_analysis,
            '5': example_llm_quick_eval,
            '6': example_semantic_similarity,
            '7': example_error_analysis,
            '8': example_cross_validation
        }
        
        if choice == '0':
            for func in examples.values():
                try:
                    func()
                except Exception as e:
                    print(f"\n⚠ Example failed: {e}")
        elif choice in examples:
            examples[choice]()
        else:
            print("Invalid choice!")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == '__main__':
    main()