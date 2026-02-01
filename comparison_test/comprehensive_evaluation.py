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

class ComprehensiveMultiModelEvaluator:
    """
    Tüm model kombinasyonlarını test eden ana framework
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
        
        # Structural evaluator (always used)
        self.structural_eval = MedicalSchemaEvaluator()
        
        print("="*70)
        print("COMPREHENSIVE MULTI-MODEL EVALUATOR")
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
            1 +  # structural only
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
        # MODE 1: Structural Only (Baseline)
        # ====================================================================
        if 'structural' in run_modes:
            print(f"\n{'='*70}")
            print("MODE 1: STRUCTURAL EVALUATION ONLY")
            print(f"{'='*70}")
            
            results = self._run_structural_only(ground_truth, sample_files)
            results_summary['results']['structural_only'] = results
            
            self._save_results(results, "structural_only")
        
        # ====================================================================
        # MODE 2: Embedding Models (without LLM)
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
        # MODE 3: LLM Models (without embeddings)
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
        # MODE 4: Combined (Embedding + LLM)
        # ====================================================================
        if 'combined' in run_modes:
            print(f"\n{'='*70}")
            print("MODE 4: COMBINED EVALUATION (Embedding + LLM)")
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
        """Sadece structural evaluation"""
        results = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                prediction = json.load(f)
            
            structural = self.structural_eval.compare_schemas(
                ground_truth,
                prediction
            )
            
            results.append({
                'sample': sample_file.name,
                'structural': structural
            })
            
            print(f"  {sample_file.name}: {structural['overall_score']:.3f}")
        
        return results
    
    def _run_with_embedding(self, emb_key, ground_truth, sample_files):
        """Structural + tek embedding model"""
        from multi_embedding_evaluator import EmbeddingEvaluator
        
        emb_config = self.EMBEDDING_MODELS[emb_key]
        emb_eval = EmbeddingEvaluator(emb_config['name'])
        
        results = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                prediction = json.load(f)
            
            # Structural
            structural = self.structural_eval.compare_schemas(
                ground_truth,
                prediction
            )
            
            # Embedding
            semantic_sim = emb_eval.compute_similarity(
                ground_truth,
                prediction
            )
            
            results.append({
                'sample': sample_file.name,
                'structural': structural,
                'semantic_similarity': semantic_sim
            })
            
            print(f"  {sample_file.name}: struct={structural['overall_score']:.3f}, sem={semantic_sim:.3f}")
        
        return results
    
    def _run_with_llm(self, llm_key, ground_truth, sample_files):
        """Structural + tek LLM model"""
        llm_config = self.LLM_MODELS[llm_key]
        
        llm_eval = LLMEvaluator(
            model_type=llm_config['type'],
            model_name=llm_config['name'],
            api_key=self.api_keys[llm_config['type']]
        )
        
        results = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                prediction = json.load(f)
            
            # Structural
            structural = self.structural_eval.compare_schemas(
                ground_truth,
                prediction
            )
            
            # LLM
            try:
                llm_result = llm_eval.evaluate_schema_pair(
                    ground_truth,
                    prediction,
                    ground_truth['input']
                )
            except Exception as e:
                llm_result = {'error': str(e)}
            
            results.append({
                'sample': sample_file.name,
                'structural': structural,
                'llm_evaluation': llm_result
            })
            
            llm_score = llm_result.get('similarity_score', 0) if 'error' not in llm_result else 0
            print(f"  {sample_file.name}: struct={structural['overall_score']:.3f}, llm={llm_score:.3f}")
        
        return results
    
    def _run_combined(self, emb_key, llm_key, ground_truth, sample_files):
        """Structural + Embedding + LLM (full pipeline)"""
        from multi_embedding_evaluator import EmbeddingEvaluator
        
        # Init evaluators
        emb_config = self.EMBEDDING_MODELS[emb_key]
        emb_eval = EmbeddingEvaluator(emb_config['name'])
        
        llm_config = self.LLM_MODELS[llm_key]
        llm_eval = LLMEvaluator(
            model_type=llm_config['type'],
            model_name=llm_config['name'],
            api_key=self.api_keys[llm_config['type']]
        )
        
        results = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                prediction = json.load(f)
            
            # Structural
            structural = self.structural_eval.compare_schemas(
                ground_truth,
                prediction
            )
            
            # Embedding
            semantic_sim = emb_eval.compute_similarity(
                ground_truth,
                prediction
            )
            
            # LLM
            try:
                llm_result = llm_eval.evaluate_schema_pair(
                    ground_truth,
                    prediction,
                    ground_truth['input']
                )
            except Exception as e:
                llm_result = {'error': str(e)}
            
            results.append({
                'sample': sample_file.name,
                'structural': structural,
                'semantic_similarity': semantic_sim,
                'llm_evaluation': llm_result
            })
            
            llm_score = llm_result.get('similarity_score', 0) if 'error' not in llm_result else 0
            print(f"  {sample_file.name}: struct={structural['overall_score']:.3f}, sem={semantic_sim:.3f}, llm={llm_score:.3f}")
        
        return results
    
    def _save_results(self, results, mode_name):
        """Save results to directory"""
        output_dir = self.output_base_dir / mode_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary
        self._write_summary(results, output_dir / "summary.txt", mode_name)
        
        print(f"    💾 Saved to: {output_dir}/")
    
    def _write_summary(self, results, output_path, mode_name):
        """Write human-readable summary"""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"EVALUATION SUMMARY: {mode_name}\n")
            f.write("="*70 + "\n\n")
            
            for r in results:
                f.write(f"{r['sample']}:\n")
                f.write(f"  Structural: {r['structural']['overall_score']:.4f}\n")
                
                if 'semantic_similarity' in r:
                    f.write(f"  Semantic: {r['semantic_similarity']:.4f}\n")
                
                if 'llm_evaluation' in r and 'similarity_score' in r['llm_evaluation']:
                    f.write(f"  LLM: {r['llm_evaluation']['similarity_score']:.4f}\n")
                
                f.write("\n")
    
    def _generate_final_report(self, results_summary):
        """Generate comprehensive comparison report"""
        report_path = self.output_base_dir / "FINAL_COMPARISON_REPORT.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE MULTI-MODEL EVALUATION - FINAL REPORT\n")
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
                
                # Calculate average scores
                struct_scores = [r['structural']['overall_score'] for r in results]
                avg_struct = sum(struct_scores) / len(struct_scores)
                
                f.write(f"{mode}:\n")
                f.write(f"  Avg Structural: {avg_struct:.4f}\n")
                
                if 'semantic_similarity' in results[0]:
                    sem_scores = [r['semantic_similarity'] for r in results]
                    avg_sem = sum(sem_scores) / len(sem_scores)
                    f.write(f"  Avg Semantic: {avg_sem:.4f}\n")
                
                if 'llm_evaluation' in results[0]:
                    llm_scores = [
                        r['llm_evaluation'].get('similarity_score', 0)
                        for r in results
                        if 'error' not in r['llm_evaluation']
                    ]
                    if llm_scores:
                        avg_llm = sum(llm_scores) / len(llm_scores)
                        f.write(f"  Avg LLM: {avg_llm:.4f}\n")
                
                f.write("\n")
        
        print(f"\n{'='*70}")
        print(f"✅ FINAL REPORT: {report_path}")
        print(f"{'='*70}")