comparison_test/
│
├── 📊 VERİ
│   ├── data/0/
│   │   ├── gt0.json              → Ground truth (doğru cevap)
│   │   └── sample0.X.json        → Modelinizin tahminleri
│   │   └── data/0_normalized/    → Otomatik düzeltilmiş versiyonlar
│
├── 🧠 CORE MODÜLLERİ (Evaluation Mekanizmaları)
│   ├── medical_schema_evaluator.py
│   │   └── Field-by-field karşılaştırma (structural)
│   │       • abnormality doğru mu? ✓/✗
│   │       • presence doğru mu? ✓/✗
│   │       • location eşleşiyor mu? ✓/✗
│   │
│   ├── multi_embedding_evaluator.py
│   │   └── Semantic benzerlik (BioBERT, PubMedBERT...)
│   │       • "fracture" ≈ "break" → 0.95 similarity
│   │       • "fracture" ≈ "effusion" → 0.30 similarity
│   │
│   └── llm_evaluator.py
│       └── LLM ile klinik validasyon (Gemini, Gemma...)
│           • "Aynı anlama mı geliyor?" → Yes/No
│           • "Kritik hata var mı?" → List
│
├── 🎛️ ORKESTRATÖRLERİ (Ana Kontrol)
│   ├── comprehensive_evaluation.py
│   │   └── Tüm modelleri test et, sonuçları karşılaştır
│   │
│   └── comprehensive_evaluation.ipynb
│       └── Notebook versiyonu (çalıştırmak için)
│
└── 📄 SONUÇLAR
    └── data/0_normalized/ulti_comp_results/
        ├── structural_only/          → Sadece rule-based
        ├── embedding_biobert/        → BioBERT semantic
        ├── llm_gemini_pro/          → LLM validation
        └── combined_X_Y/            → Hepsi birlikte


        

INPUT: 
  GT: "fracture, present, distal radius, acute, 5mm"
  Prediction: "break, present, distal radius, sharp, about 5mm"

↓

LEVEL 1: STRUCTURAL (medical_schema_evaluator.py)
  ├─ abnormality: "fracture" vs "break" → FARKLI (0.0)
  ├─ presence: "present" vs "present" → AYNI (1.0)
  ├─ location: "distal radius" vs "distal radius" → AYNI (1.0)
  ├─ degree: "acute" vs "sharp" → FARKLI (0.0)
  └─ measurement: "5mm" vs "about 5mm" → YAKIN (0.7)
  
  SKOR: 0.54 (weighted average)

↓

LEVEL 2: SEMANTIC (multi_embedding_evaluator.py)
  Text1: "fracture present distal radius acute 5mm"
  Text2: "break present distal radius sharp about 5mm"
  
  BioBERT embedding → Cosine similarity: 0.98
  (Çünkü "fracture"≈"break", "acute"≈"sharp" synonym)

↓

LEVEL 3: LLM (llm_evaluator.py)
  Gemini'ye sor: "Klinik olarak aynı mı?"
  
  Cevap: {
    "similarity": 0.7,
    "clinical_equivalence": "high",
    "critical_errors": ["degree mismatch"],
    "assessment": "Semantically equivalent, minor terminology difference"
  }

↓

FINAL REPORT:
  Structural: 0.54 (katı kurallar)
  Semantic: 0.98 (synonym yakalar)
  LLM: 0.70 (klinik bakış)
  
  SONUÇ: Model iyi çalışıyor ama terminology standardize edilmeli
1. comprehensive_evaluation.ipynb açılır
   ↓
2. CELL 0: JSON normalization (otomatik düzeltme)
   ↓
3. CELL 1: Evaluation başlat
   ↓
4. Loop: Her sample için
   ├─ Structural eval
   ├─ Embedding eval (4 farklı model)
   └─ LLM eval (Gemini)
   ↓
5. Sonuçları kaydet:
   ├─ JSON (detaylı)
   ├─ TXT (özet)
   └─ FINAL_REPORT.txt (karşılaştırma)



   
# Medical Schema Extraction - Evaluation Framework

A comprehensive, multi-level evaluation framework for medical information extraction systems, designed for academic research and publication.

## 🎯 Overview

This framework evaluates medical schema extraction models using three complementary approaches:

1. **Structural Evaluation** - Rule-based field-by-field comparison
2. **Semantic Evaluation** - Embedding-based similarity using BioBERT
3. **Clinical Evaluation** - LLM-based assessment of clinical equivalence

## 🌟 Key Features

- ✅ **Duplicate Detection** - Prevents data leakage in train/test splits
- ✅ **Stratified Splitting** - Groups by input to avoid information leakage
- ✅ **Multi-Level Scoring** - Weighted field importance for clinical relevance
- ✅ **Statistical Validation** - Bootstrap confidence intervals & significance tests
- ✅ **LLM Integration** - Uses free APIs (Gemini, HuggingFace, Together AI)
- ✅ **Comprehensive Reporting** - JSON + human-readable summaries

## 📋 Schema Format

Input schema format (same as schema_train.json):

```json
{
  "instruction": "Extract medical entities...",
  "input": "No fracture observed in the skull.",
  "output": [
    {
      "abnormality": "fracture",
      "finding": "None",
      "presence": "absent",
      "location": ["skull"],
      "degree": "None",
      "measurement": "None",
      "comparison": "None"
    }
  ]
}
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the framework
cd medical-schema-evaluation

# Install dependencies
pip install -r requirements.txt

# Install sentence-transformers for semantic evaluation
pip install sentence-transformers
```

### 2. Setup API Keys (Optional but Recommended)

For LLM-based evaluation, get a free API key:

**Option A: Google Gemini (Recommended - 2M tokens/day FREE)**
```bash
# Get key from: https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="your-api-key-here"
```

**Option B: HuggingFace**
```bash
# Get key from: https://huggingface.co/settings/tokens
export HUGGINGFACE_API_KEY="your-api-key-here"
```

**Option C: Together AI**
```bash
# Get key from: https://api.together.xyz/
export TOGETHER_API_KEY="your-api-key-here"
```

### 3. Prepare Your Data

You need two files:

1. **Ground Truth**: `schema_train.json` (your uploaded file)
2. **Predictions**: Your model's output in the same format

Example prediction file (`model_predictions.json`):
```json
[
  {
    "instruction": "Extract medical entities...",
    "input": "No fracture observed in the skull.",
    "output": [
      {
        "abnormality": "fracture",
        "finding": "None",
        "presence": "absent",
        "location": ["cranial bones"],
        "degree": "None",
        "measurement": "None",
        "comparison": "None"
      }
    ]
  }
]
```

### 4. Run Evaluation

```python
from evaluation_pipeline import ComprehensiveEvaluationPipeline

# Initialize pipeline
pipeline = ComprehensiveEvaluationPipeline(
    use_llm=True,           # Enable LLM evaluation
    use_embeddings=True,    # Enable semantic similarity
    llm_provider='gemini'   # or 'huggingface' or 'together'
)

# Run complete evaluation
results = pipeline.run_complete_evaluation(
    ground_truth_path='schema_train.json',
    predictions_path='model_predictions.json',
    output_dir='./evaluation_results',
    test_size=0.2,          # 20% for testing
    random_seed=42          # For reproducibility
)

# Results are automatically saved to output_dir
```

Or run from command line:
```bash
python evaluation_pipeline.py
```

## 📊 Evaluation Metrics

### Structural Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **Field-wise Accuracy**: Individual accuracy for each field
  - `abnormality` (weight: 0.25)
  - `presence` (weight: 0.25) - Critical for clinical decisions
  - `location` (weight: 0.20)
  - `degree` (weight: 0.15)
  - `measurement` (weight: 0.10)
- **Exact Match Rate**: Percentage of perfect predictions

### Semantic Metrics
- **Cosine Similarity**: Using BioBERT embeddings
- Captures synonyms and paraphrasing

### Clinical Metrics (LLM-based)
- **Clinical Equivalence**: High/Medium/Low/None
- **Clinical Impact**: Critical/Moderate/Minor/None
- **Detailed Explanations**: Per-entity analysis

### Statistical Metrics
- **Bootstrap Confidence Intervals** (95%)
- **Paired t-tests** for model comparison
- **Cohen's d** effect size

## 📁 Output Files

After running evaluation, you'll get:

```
evaluation_results/
├── evaluation_report_TIMESTAMP.json    # Main results
├── structural_results_TIMESTAMP.json  # Detailed scores
├── llm_results_TIMESTAMP.json         # LLM evaluations
├── summary_TIMESTAMP.txt              # Human-readable
└── duplicates_TIMESTAMP.json          # Duplicate analysis
```

## 🔬 Academic Usage

### For Your Paper

This framework provides:

1. **Rigorous Methodology**
   - Proper train/test splitting with no leakage
   - Statistical significance testing
   - Multiple evaluation perspectives

2. **Reproducibility**
   - Fixed random seeds
   - Documented configurations
   - Complete error analysis

3. **Comprehensive Metrics**
   - Rule-based (objective)
   - Embedding-based (semantic)
   - LLM-based (clinical validity)

### Citation-Worthy Features

- **Multi-level Evaluation**: Novel combination of rule-based, semantic, and LLM approaches
- **Clinical Weighting**: Field importance based on clinical relevance
- **Duplicate-Aware Splitting**: Prevents overoptimistic performance estimates
- **Free & Reproducible**: All tools are free and open-source

### Suggested Methodology Section

```
We evaluated our model using a comprehensive three-level framework:

1. Structural Evaluation: Rule-based field-by-field comparison with 
   clinically-weighted importance scores.

2. Semantic Evaluation: BioBERT embeddings (dmis-lab/biobert-base-cased-v1.2) 
   to capture semantic similarity and handle paraphrasing.

3. Clinical Validation: LLM-based assessment using Gemini 1.5 Flash to 
   determine clinical equivalence and identify critical vs. minor errors.

Data splitting employed stratification by input text to prevent information 
leakage from duplicate samples. Statistical significance was established using 
bootstrap confidence intervals (10,000 iterations) and paired t-tests.
```

## 🧪 Advanced Usage

### Custom Field Weights

```python
from medical_schema_evaluator import MedicalSchemaEvaluator

evaluator = MedicalSchemaEvaluator()

# Customize weights based on your clinical focus
evaluator.FIELD_WEIGHTS = {
    'abnormality': 0.30,  # Increase if abnormality detection is critical
    'presence': 0.30,
    'location': 0.20,
    'degree': 0.10,
    'measurement': 0.10,
}
```

### Statistical Comparison of Models

```python
from medical_schema_evaluator import StatisticalAnalyzer

# Compare two models
baseline_scores = [0.85, 0.82, 0.88, ...]  # Model A
new_model_scores = [0.89, 0.87, 0.91, ...]  # Model B

comparison = StatisticalAnalyzer.compare_models(
    baseline_scores,
    new_model_scores
)

print(f"Improvement: {comparison['improvement']:.4f}")
print(f"P-value: {comparison['p_value']:.4f}")
print(f"Significant: {comparison['significant']}")
```

### Custom LLM Evaluation Prompt

```python
from llm_evaluator import LLMEvaluator

evaluator = LLMEvaluator(provider='gemini')

# Modify the prompt in llm_evaluator.py:_build_evaluation_prompt()
# to focus on specific aspects relevant to your research
```

## 🐛 Troubleshooting

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "API key not found"
```bash
# Set environment variable
export GEMINI_API_KEY="your-key"

# Or set in Python
import os
os.environ['GEMINI_API_KEY'] = 'your-key'
```

### "Rate limit exceeded"
- Use `time.sleep()` between API calls
- Reduce batch size in `batch_evaluate()`
- Switch to a different provider

### "Model download failed"
```python
# Download models manually
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

## 📈 Performance Optimization

### For Large Datasets

```python
# 1. Sample for LLM evaluation (most expensive)
sample_size = 50  # Instead of evaluating all

# 2. Disable LLM for initial testing
pipeline = ComprehensiveEvaluationPipeline(use_llm=False)

# 3. Cache embeddings
# Embeddings are computed once and can be reused

# 4. Parallel processing (future enhancement)
# Can parallelize structural evaluation across samples
```

## 🤝 Contributing

This is a research framework. Suggestions welcome:

- Additional evaluation metrics
- Support for more LLM providers
- Visualization tools
- Cross-validation strategies

## 📚 References

### Models & APIs Used

- **BioBERT**: Lee et al. (2020), "BioBERT: a pre-trained biomedical language representation model"
- **Sentence-BERT**: Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **Gemini 1.5**: Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models"

### Evaluation Methodology

- Bootstrap confidence intervals: Efron & Tibshirani (1993)
- Cohen's d effect size: Cohen (1988)
- Inter-annotator agreement: Krippendorff's alpha

## 📝 License

This framework is provided for research and academic use.

## 🙋 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the example code
3. Consult the inline documentation in source files

---

**Good luck with your research! 🚀**
