# ============================================================================
# MULTI-MODEL EVALUATION FRAMEWORK
# ============================================================================
# comprehensive_evaluation.py - ANA ORCHESTRATOR

import json
import os
from pathlib import Path
from datetime import datetime
import itertools
import logging
import transformers
transformers.logging.set_verbosity_error()  # Sadece hataları göster, uyarıları gizle
# Mevcut modüller
from medical_schema_evaluator import MedicalSchemaEvaluator
from llm_evaluator import LLMEvaluator

# Yeni modül (aşağıda oluşturacağız)
from multi_embedding_evaluator import EmbeddingEvaluator
# comprehensive_evaluation.py (NEW SECTION)

class StandardizedOutputFormatter:
    """
    Ensures consistent output format across all evaluation modes
    """
    
    @staticmethod
    def format_evaluation_result(
        test_file: str,
        gt_file: str,
        alignment_info: Dict,
        entity_comparisons: List[Dict],
        model_scores: Dict,
        aggregates: Dict
    ) -> Dict:
        """
        Standard output format for ALL evaluations
        
        Returns:
            {
                "metadata": {...},
                "alignment": {...},
                "entity_level": {...},
                "model_scores": {...},
                "aggregates": {...}
            }
        """
        
        return {
            "metadata": {
                "test_file": test_file,
                "gt_file": gt_file,
                "timestamp": datetime.now().isoformat(),
                "evaluation_version": "2.0"
            },
            
            "report_validation": {
                "reports_match": alignment_info['reports_match'],
                "report_similarity": alignment_info.get('report_similarity', 1.0)
            },
            
            "alignment": {
                "strategy_used": alignment_info['strategy'],
                "input_level": {
                    "gt_inputs": alignment_info['gt_input_count'],
                    "pred_inputs": alignment_info['pred_input_count'],
                    "exact_matches": alignment_info['exact_matches_count'],
                    "fuzzy_matches": alignment_info['fuzzy_matches_count'],
                    "match_rate": alignment_info['match_rate'],
                    "quality": alignment_info['alignment_quality']
                }
            },
            
            "entity_level": {
                "total_gt_entities": len([e for comp in entity_comparisons for e in comp['gt_entities']]),
                "total_pred_entities": len([e for comp in entity_comparisons for e in comp['pred_entities']]),
                "matched_entities": sum(comp['matched_count'] for comp in entity_comparisons),
                "precision": aggregates.get('precision', 0.0),
                "recall": aggregates.get('recall', 0.0),
                "f1_score": aggregates.get('f1_score', 0.0),
                "field_wise_scores": aggregates.get('field_wise', {})
            },
            
            "model_scores": {
                "structural": {
                    "mean": aggregates.get('structural_mean', 0.0),
                    "std": aggregates.get('structural_std', 0.0),
                    "details": model_scores.get('structural', [])
                },
                "embeddings": {
                    model_name: {
                        "mean": aggregates.get(f'{model_name}_mean', 0.0),
                        "std": aggregates.get(f'{model_name}_std', 0.0),
                        "details": model_scores.get(model_name, [])
                    }
                    for model_name in model_scores.keys() if model_name.startswith('embedding_')
                },
                "llm": {
                    model_name: {
                        "mean": aggregates.get(f'{model_name}_mean', 0.0),
                        "std": aggregates.get(f'{model_name}_std', 0.0),
                        "details": model_scores.get(model_name, [])
                    }
                    for model_name in model_scores.keys() if model_name.startswith('llm_')
                }
            },
            
            "detailed_comparisons": entity_comparisons
        }

#=============================================================================
# MAIN UPDATED CLASS: ComprehensiveMultiModelEvaluator
# =============================================================================

class ComprehensiveMultiModelEvaluator:
    """
    Tüm model kombinasyonlarını test eden ana framework
    Now with Entity-Level Evaluation (FIX 3 applied)
    """
    
    # ========================================================================
    # MODEL CONFIGURATIONS
    # ========================================================================
    
    LLM_MODELS = {
        "gemini_flash": {
            "type": "gemini",
            "name": "models/gemini-2.5-flash",
            "description": "Hızlı, free tier"
        },
        "gemini_pro": {
            "type": "gemini",
            "name": "models/gemini-2.5-pro",
            "description": "Yüksek kalite"
        },
        "gemini_flash_2": {
            "type": "gemini",
            "name": "gemini-2.5-flash",
            "description": "En yeni, hızlı"
        },
        "gemma": {
            "type": "gemma",
            "name": "gemma-3-27b-it",
            "description": "Açık kaynak"
        },
        "glm": {
            "type": "glm",
            "name": "glm-4-flash",
            "description": "ZhipuAI"
        },
        "deepseek": {
            "type": "deepseek",
            "name": "deepseek-chat",
            "description": "DeepSeek"
        }
    }
    
    EMBEDDING_MODELS = {
        "biobert": {
            "name": "dmis-lab/biobert-v1.1",
            "description": "PubMed/MEDLINE fine-tuned",
            "quality": "ÇÖP",
            "speed": "Fast"
        },
        "pubmedbert": {
            "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "description": "PubMed from scratch",
            "quality": "⭐⭐⭐⭐⭐",
            "speed": "Fast"
        },
        "clinicalbert": {
            "name": "emilyalsentzer/Bio_ClinicalBERT",
            "description": "ÇÖP",
            "quality": "⭐⭐⭐⭐⭐",
            "speed": "Fast"
        },
        "sapbert": {
            "name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            "description": "Synonym-aware, UMLS",
            "quality": "⭐⭐⭐⭐⭐",
            "speed": "Medium"
        },
        "bluebert": {
            "name": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
            "description": "PubMed + MIMIC hybrid",
            "quality": "⭐⭐⭐⭐",
            "speed": "Fast"
        },
        "s_pubmedbert": {
            "name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "description": "Sentence-level fine-tuned",
            "quality": "GEMİNİ İYİ OLDUĞUNU SÖYLÜYOR",
            "speed": "Fast"
        },
        "general_baseline": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "General baseline",
            "quality": "GEMİNİ İYİ OLDUĞUNU SÖYLÜYOR",
            "speed": "Very Fast"
        },
        "general_mpnet": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "description": "General high-quality",
            "quality": "⭐⭐⭐⭐",
            "speed": "Medium"
        },
        "neuml_pubmedbert": {
            "name": "NeuML/pubmedbert-base-embeddings",
            "description": "GEMİNİ İYİ OLDUĞUNU SÖYLÜYOR",
            "quality": "⭐⭐⭐⭐",
            "speed": "Fast"
        }
    }
    
    def __init__(self, 
                 api_keys,
                 data_dir,
                 output_base_dir,
                 selected_llms=None,
                 selected_embeddings=None):
        """
        Args:
            api_keys: Dict of API keys
            data_dir: Data directory (e.g., "./data/0/")
            output_base_dir: Base results directory
            selected_llms: List of LLM keys to test (None = all)
            selected_embeddings: List of embedding keys to test (None = all)
        """
        self.api_keys = api_keys
        self.data_dir = Path(data_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # Model selection
        self.selected_llms = selected_llms or list(self.LLM_MODELS.keys())
        self.selected_embeddings = selected_embeddings or list(self.EMBEDDING_MODELS.keys())
        
        from entity_level_evaluator import EntityLevelEvaluator
        self.entity_evaluator = EntityLevelEvaluator(
            use_semantic_matching=use_semantic_matching,
            use_llm_for_borderline=False,  # İsterseniz True yapıp llm_evaluator verebilirsiniz
            llm_evaluator=None
        )
        
        print("="*70)
        print("COMPREHENSIVE MULTI-MODEL EVALUATOR v2.1 (Semantic Entity Matching)")
        print("="*70)
        print(f"\nData directory: {self.data_dir}")
        print(f"Output directory: {self.output_base_dir}")
        print(f"\nLLM models to test: {len(self.selected_llms)}")
        for llm in self.selected_llms:
            print(f"  - {llm}: {self.LLM_MODELS[llm]['name']}")
        print(f"\nEmbedding models to test: {len(self.selected_embeddings)}")
        for emb in self.selected_embeddings:
            print(f"  - {emb}: {self.EMBEDDING_MODELS[emb]['name']}")
        
        total_combinations = (
            1 +  # structural only (now entity-level)
            len(self.selected_embeddings) +  # embedding only
            len(self.selected_llms) +  # LLM only
            len(self.selected_embeddings) * len(self.selected_llms)  # combined
        )
        print(f"\n⚠️  Total combinations: {total_combinations}")
        print("="*70)
    
    def run_full_evaluation(self, gt_file, run_modes=None):
        """
        Tüm kombinasyonları test et
        
        Args:
            gt_file: Ground truth filename
            run_modes: List of modes to run:
                      ['structural', 'embedding', 'llm', 'combined']
                      None = all modes
        """
        if run_modes is None:
            run_modes = ['structural', 'embedding', 'llm', 'combined']
        
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': str(self.data_dir),
            'gt_file': gt_file,
            'modes_run': run_modes,
            'results': {}
        }
        
        # Load data
        gt_path = self.data_dir / gt_file
        sample_files = sorted(self.data_dir.glob("sample*.json"))
        
        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"LOADING DATA")
        print(f"{'='*70}")
        print(f"Ground truth: {gt_file}")
        print(f"Samples: {len(sample_files)}")
        
        # ====================================================================
        # MODE 1: Entity-Level Structural Only (NEW - replaces old structural)
        # ====================================================================
        if 'structural' in run_modes:
            print(f"\n{'='*70}")
            print("MODE 1: ENTITY-LEVEL STRUCTURAL EVALUATION")
            print(f"{'='*70}")
            
            results = self._run_structural_only(ground_truth, sample_files)
            results_summary['results']['structural_only'] = results
            
            self._save_results(results, "structural_only")
        
        # ====================================================================
        # MODE 2: Embedding Models (Entity-Level + Embeddings)
        # ====================================================================
        if 'embedding' in run_modes:
            print(f"\n{'='*70}")
            print("MODE 2: EMBEDDING MODELS EVALUATION")
            print(f"{'='*70}")
            
            for emb_key in self.selected_embeddings:
                print(f"\n--- Testing: {emb_key} ---")
                
                results = self._run_with_embedding(
                    emb_key,
                    ground_truth,
                    sample_files
                )
                
                results_summary['results'][f'embedding_{emb_key}'] = results
                self._save_results(results, f"embedding_{emb_key}")
        
        # ====================================================================
        # MODE 3: LLM Models (Entity-Level + LLM)
        # ====================================================================
        if 'llm' in run_modes:
            print(f"\n{'='*70}")
            print("MODE 3: LLM MODELS EVALUATION")
            print(f"{'='*70}")
            
            for llm_key in self.selected_llms:
                print(f"\n--- Testing: {llm_key} ---")
                
                results = self._run_with_llm(
                    llm_key,
                    ground_truth,
                    sample_files
                )
                
                results_summary['results'][f'llm_{llm_key}'] = results
                self._save_results(results, f"llm_{llm_key}")
        
        # ====================================================================
        # MODE 4: Combined (Entity-Level + Embedding + LLM)
        # ====================================================================
        if 'combined' in run_modes:
            print(f"\n{'='*70}")
            print("MODE 4: COMBINED EVALUATION (Entity-Level + Embedding + LLM)")
            print(f"{'='*70}")
            
            combinations = list(itertools.product(
                self.selected_embeddings,
                self.selected_llms
            ))
            
            print(f"Total combinations: {len(combinations)}")
            
            for idx, (emb_key, llm_key) in enumerate(combinations, 1):
                print(f"\n[{idx}/{len(combinations)}] {emb_key} + {llm_key}")
                
                results = self._run_combined(
                    emb_key,
                    llm_key,
                    ground_truth,
                    sample_files
                )
                
                combo_key = f"combined_{emb_key}_{llm_key}"
                results_summary['results'][combo_key] = results
                self._save_results(results, combo_key)
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        self._generate_final_report(results_summary)
        
        return results_summary
    
    def _run_structural_only(self, ground_truth, sample_files):
        """Entity-level structural evaluation (replaces old structural)"""
        results = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                prediction = json.load(f)
            
            # Validate reports match (essential for entity-level strategy)
            gt_report = ground_truth.get('report', '')
            pred_report = prediction.get('report', '')
            
            if gt_report != pred_report:
                print(f"  ⚠️  {sample_file.name}: Reports don't match! Skipping.")
                continue
            
            gt_entities = self.entity_evaluator.flatten_entities(ground_truth)
            pred_entities = self.entity_evaluator.flatten_entities(prediction)
            
            print(f"  📊 Matching {len(gt_entities)} GT entities with {len(pred_entities)} Pred entities...")
            
            # Match with semantic similarity
            entity_matches = self.entity_evaluator.match_entities(
                gt_entities, 
                pred_entities,
                report_text=gt_report  # Context for LLM (if used)
            )
            
            # Compute metrics
            entity_metrics = self.entity_evaluator.compute_metrics(entity_matches)
            
            # Detailed output
            print(f"  ✅ Results:")
            print(f"     Precision: {entity_metrics['precision']:.3f}")
            print(f"     Recall:    {entity_metrics['recall']:.3f}")
            print(f"     F1-Score:  {entity_metrics['f1_score']:.3f}")
            print(f"     Matches:   {entity_metrics['true_positives']} TP, "
                  f"{entity_metrics['false_positives']} FP, "
                  f"{entity_metrics['false_negatives']} FN")
            
            if entity_metrics['contradiction_count'] > 0:
                print(f"     ⚠️  Contradictions detected: {entity_metrics['contradiction_count']}")
            
            # Format result (standardized)
            result = {
                'metadata': {
                    'test_file': sample_file.name,
                    'gt_file': 'ground_truth.json',
                    'timestamp': datetime.now().isoformat()
                },
                'entity_level': {
                    'total_gt_entities': entity_metrics['total_gt_entities'],
                    'total_pred_entities': entity_metrics['total_pred_entities'],
                    'matched_entities': entity_metrics['true_positives'],
                    'precision': entity_metrics['precision'],
                    'recall': entity_metrics['recall'],
                    'f1_score': entity_metrics['f1_score'],
                    'avg_match_quality': entity_metrics['avg_match_quality'],
                    'contradictions': entity_metrics['contradiction_count']
                },
                'detailed_matches': entity_matches  # Her bir eşleşmenin detayı
            }
            
            results.append(result)
        
        return results
    
    def _run_with_embedding(self, emb_key, ground_truth, sample_files):
        """Entity-Level + Embedding evaluation"""
        from multi_embedding_evaluator import EmbeddingEvaluator
        
        emb_config = self.EMBEDDING_MODELS[emb_key]
        emb_eval = EmbeddingEvaluator(emb_config['name'])
        
        results = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                prediction = json.load(f)
            
            # Entity-level evaluation first
            gt_entities = self.entity_evaluator.flatten_entities(ground_truth)
            pred_entities = self.entity_evaluator.flatten_entities(prediction)
            entity_matches = self.entity_evaluator.match_entities(gt_entities, pred_entities)
            entity_metrics = self.entity_evaluator.compute_metrics(entity_matches)
            
            # Embedding evaluation on matched entities
            embedding_scores = self._evaluate_with_embeddings(entity_matches, emb_eval)
            
            # Format result
            result = StandardizedOutputFormatter.format_evaluation_result(
                test_file=sample_file.name,
                gt_file="ground_truth.json",
                alignment_info={
                    'reports_match': True,
                    'strategy': 'entity_level_with_embedding',
                    'gt_entity_count': len(gt_entities),
                    'pred_entity_count': len(pred_entities),
                },
                entity_comparisons=entity_matches,
                model_scores={
                    'structural': entity_metrics,
                    f'embedding_{emb_key}': embedding_scores,
                },
                aggregates={
                    **entity_metrics,
                    'embedding_mean': np.mean(embedding_scores) if embedding_scores else 0.0
                }
            )
            
            results.append(result)
            
            emb_mean = np.mean(embedding_scores) if embedding_scores else 0.0
            print(f"  {sample_file.name}: F1={entity_metrics['f1_score']:.3f}, "
                  f"Emb={emb_mean:.3f}")
        
        return results
    
    def _run_with_llm(self, llm_key, ground_truth, sample_files):
        """Entity-Level + LLM evaluation"""
        llm_config = self.LLM_MODELS[llm_key]
        
        # Assuming LLMEvaluator is imported/available
        # If you don't have this class, implement similarly to embeddings
        try:
            llm_eval = LLMEvaluator(
                model_type=llm_config['type'],
                model_name=llm_config['name'],
                api_key=self.api_keys[llm_config['type']]
            )
        except:
            llm_eval = None
        
        results = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                prediction = json.load(f)
            
            # Entity-level evaluation
            gt_entities = self.entity_evaluator.flatten_entities(ground_truth)
            pred_entities = self.entity_evaluator.flatten_entities(prediction)
            entity_matches = self.entity_evaluator.match_entities(gt_entities, pred_entities)
            entity_metrics = self.entity_evaluator.compute_metrics(entity_matches)
            
            # LLM evaluation on matched entities (if available)
            llm_scores = []
            if llm_eval:
                try:
                    # Evaluate on first few matched pairs to save API calls
                    llm_result = llm_eval.evaluate_entity_matches(
                        entity_matches[:5],  # Limit to avoid too many API calls
                        ground_truth.get('report', '')
                    )
                    llm_scores = [llm_result.get('score', 0.0)]
                except Exception as e:
                    print(f"    LLM Error: {e}")
                    llm_scores = []
            
            # Format result
            result = StandardizedOutputFormatter.format_evaluation_result(
                test_file=sample_file.name,
                gt_file="ground_truth.json",
                alignment_info={
                    'reports_match': True,
                    'strategy': 'entity_level_with_llm',
                    'gt_entity_count': len(gt_entities),
                    'pred_entity_count': len(pred_entities),
                },
                entity_comparisons=entity_matches,
                model_scores={
                    'structural': entity_metrics,
                    f'llm_{llm_key}': llm_scores,
                },
                aggregates={
                    **entity_metrics,
                    'llm_mean': np.mean(llm_scores) if llm_scores else 0.0
                }
            )
            
            results.append(result)
            
            llm_mean = np.mean(llm_scores) if llm_scores else 0.0
            print(f"  {sample_file.name}: F1={entity_metrics['f1_score']:.3f}, "
                  f"LLM={llm_mean:.3f}")
        
        return results
    
    def _run_combined(self, emb_key, llm_key, ground_truth, sample_files):
        """Entity-Level + Embedding + LLM (full pipeline)"""
        from multi_embedding_evaluator import EmbeddingEvaluator
        
        # Init evaluators
        emb_config = self.EMBEDDING_MODELS[emb_key]
        emb_eval = EmbeddingEvaluator(emb_config['name'])
        
        llm_config = self.LLM_MODELS[llm_key]
        try:
            llm_eval = LLMEvaluator(
                model_type=llm_config['type'],
                model_name=llm_config['name'],
                api_key=self.api_keys[llm_config['type']]
            )
        except:
            llm_eval = None
        
        results = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                prediction = json.load(f)
            
            # Entity-level evaluation (always)
            gt_entities = self.entity_evaluator.flatten_entities(ground_truth)
            pred_entities = self.entity_evaluator.flatten_entities(prediction)
            entity_matches = self.entity_evaluator.match_entities(gt_entities, pred_entities)
            entity_metrics = self.entity_evaluator.compute_metrics(entity_matches)
            
            # Embedding
            embedding_scores = self._evaluate_with_embeddings(entity_matches, emb_eval)
            
            # LLM
            llm_scores = []
            if llm_eval:
                try:
                    llm_result = llm_eval.evaluate_entity_matches(
                        entity_matches[:5],
                        ground_truth.get('report', '')
                    )
                    llm_scores = [llm_result.get('score', 0.0)]
                except Exception as e:
                    llm_scores = []
            
            # Format result
            result = StandardizedOutputFormatter.format_evaluation_result(
                test_file=sample_file.name,
                gt_file="ground_truth.json",
                alignment_info={
                    'reports_match': True,
                    'strategy': 'entity_level_combined',
                    'gt_entity_count': len(gt_entities),
                    'pred_entity_count': len(pred_entities),
                },
                entity_comparisons=entity_matches,
                model_scores={
                    'structural': entity_metrics,
                    f'embedding_{emb_key}': embedding_scores,
                    f'llm_{llm_key}': llm_scores,
                },
                aggregates={
                    **entity_metrics,
                    'embedding_mean': np.mean(embedding_scores) if embedding_scores else 0.0,
                    'llm_mean': np.mean(llm_scores) if llm_scores else 0.0
                }
            )
            
            results.append(result)
            
            emb_mean = np.mean(embedding_scores) if embedding_scores else 0.0
            llm_mean = np.mean(llm_scores) if llm_scores else 0.0
            print(f"  {sample_file.name}: F1={entity_metrics['f1_score']:.3f}, "
                  f"Emb={emb_mean:.3f}, LLM={llm_mean:.3f}")
        
        return results
    
    def _evaluate_with_embeddings(self, entity_matches, emb_eval):
        """
        Compute embedding similarity on matched entity pairs
        Returns list of scores
        """
        scores = []
        
        for match in entity_matches:
            if match['match_type'] == 'matched' and match['gt_entity'] and match['pred_entity']:
                try:
                    # Convert entities to text for embedding comparison
                    gt_text = json.dumps(match['gt_entity'], ensure_ascii=False)
                    pred_text = json.dumps(match['pred_entity'], ensure_ascii=False)
                    
                    # Use your existing EmbeddingEvaluator here
                    # This is a simplified version - adjust based on your actual evaluator
                    score = emb_eval.compute_text_similarity(gt_text, pred_text)
                    scores.append(score)
                except:
                    # If individual comparison fails, skip
                    continue
        
        return scores
    
    def _save_results(self, results, mode_name):
        """Save results to directory (updated for new format)"""
        output_dir = self.output_base_dir / mode_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed JSON results
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save human-readable summary
        self._write_summary(results, output_dir / "summary.txt", mode_name)
        
        print(f"    💾 Saved to: {output_dir}/")
    
    def _write_summary(self, results, output_path, mode_name):
        """Write human-readable summary (updated for entity-level metrics)"""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"EVALUATION SUMMARY: {mode_name}\n")
            f.write("="*70 + "\n\n")
            
            for r in results:
                f.write(f"{r['metadata']['test_file']}:\n")
                
                # Entity-level metrics (primary)
                ent = r['entity_level']
                f.write(f"  Precision: {ent['precision']:.4f}\n")
                f.write(f"  Recall:    {ent['recall']:.4f}\n")
                f.write(f"  F1-Score:  {ent['f1_score']:.4f}\n")
                f.write(f"  Entities:  GT={ent['total_gt_entities']}, "
                       f"Pred={ent['total_pred_entities']}, "
                       f"Matched={ent['matched_entities']}\n")
                
                # Model scores
                if 'model_scores' in r:
                    ms = r['model_scores']
                    
                    # Embeddings
                    for emb_key, emb_scores in ms.get('embeddings', {}).items():
                        if isinstance(emb_scores, dict) and 'mean' in emb_scores:
                            f.write(f"  {emb_key}: {emb_scores['mean']:.4f}\n")
                    
                    # LLM
                    for llm_key, llm_scores in ms.get('llm', {}).items():
                        if isinstance(llm_scores, dict) and 'mean' in llm_scores:
                            f.write(f"  {llm_key}: {llm_scores['mean']:.4f}\n")
                
                f.write("\n")
    
    def _generate_final_report(self, results_summary):
        """Generate comprehensive comparison report"""
        report_path = self.output_base_dir / "FINAL_COMPARISON_REPORT.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE MULTI-MODEL EVALUATION - FINAL REPORT\n")
            f.write("Entity-Level Evaluation v2.0\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Timestamp: {results_summary['timestamp']}\n")
            f.write(f"Ground Truth: {results_summary['gt_file']}\n")
            f.write(f"Modes Run: {', '.join(results_summary['modes_run'])}\n\n")
            
            # Compare all modes
            f.write("COMPARISON ACROSS ALL MODES:\n")
            f.write("-"*70 + "\n\n")
            
            for mode, results in results_summary['results'].items():
                if not results:
                    continue
                
                f.write(f"{mode}:\n")
                
                # Calculate average entity-level metrics
                precisions = [r['entity_level']['precision'] for r in results]
                recalls = [r['entity_level']['recall'] for r in results]
                f1_scores = [r['entity_level']['f1_score'] for r in results]
                
                f.write(f"  Avg Precision: {sum(precisions)/len(precisions):.4f}\n")
                f.write(f"  Avg Recall:    {sum(recalls)/len(recalls):.4f}\n")
                f.write(f"  Avg F1-Score:  {sum(f1_scores)/len(f1_scores):.4f}\n")
                
                # Embedding scores if present
                if results[0].get('model_scores', {}).get('embeddings'):
                    for emb_key in results[0]['model_scores']['embeddings'].keys():
                        emb_means = []
                        for r in results:
                            emb_data = r['model_scores']['embeddings'].get(emb_key, {})
                            if isinstance(emb_data, dict) and 'mean' in emb_data:
                                emb_means.append(emb_data['mean'])
                        if emb_means:
                            f.write(f"  Avg {emb_key}: {sum(emb_means)/len(emb_means):.4f}\n")
                
                # LLM scores if present
                if results[0].get('model_scores', {}).get('llm'):
                    for llm_key in results[0]['model_scores']['llm'].keys():
                        llm_means = []
                        for r in results:
                            llm_data = r['model_scores']['llm'].get(llm_key, {})
                            if isinstance(llm_data, dict) and 'mean' in llm_data:
                                llm_means.append(llm_data['mean'])
                        if llm_means:
                            f.write(f"  Avg {llm_key}: {sum(llm_means)/len(llm_means):.4f}\n")
                
                f.write("\n")
        
        print(f"\n{'='*70}")
        print(f"✅ FINAL REPORT: {report_path}")
        print(f"{'='*70}")
