"""
LLM-Based Schema Evaluator
===========================
Basit ve anlaşılır API kullanımı ile medikal şema değerlendirme

Desteklenen modeller:
- Gemini (Google AI Studio)
- Gemma (Google AI Studio) 
- GLM (ZhipuAI)
- DeepSeek
- ve daha fazlası...

KULLANIM:
    # API keylerini tanımla
    API_KEYS = {
        "gemini": "AIzaSy...",
        "gemma": "KGAT_...",
        "glm": "sk-...",
        "deepseek": "sk-..."
    }
    
    # Evaluator oluştur
    evaluator = LLMEvaluator(
        model_type="gemini",
        model_name="gemini-1.5-flash",
        api_key=API_KEYS["gemini"]
    )
    
    # Şemaları değerlendir
    result = evaluator.evaluate_schema_pair(ground_truth, prediction, input_text)
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
import logging
from google.genai import types
# Google AI için
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ google-genai kurulu değil. Lütfen çalıştır:")
    print("  pip install google-genai")

# OpenAI-compatible API'ler için
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠ openai kurulu değil. GLM/DeepSeek kullanmak için:")
    print("  pip install openai")

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """
    Basit ve anlaşılır LLM-based şema değerlendirici
    
    Bu class direkt olarak API'leri kullanır, karmaşık config yok!
    """
    
    # Her model için rate limit ayarları (saniye cinsinden bekleme süresi)
    RATE_LIMITS = {
        "gemini-2.5-pro": 60 / 1,          # 1 req/min
        "gemini-2.5-flash": 60 / 5,      # 5 req/min = 12 sn bekle
        "gemini-2.5-flash-lite": 60 / 10, # 10 req/min = 6 sn bekle
        "gemini-1.5-flash": 60 / 15,      # 15 req/min = 4 sn bekle
        "gemini-1.5-pro": 60 / 2,         # 2 req/min = 30 sn bekle
        "gemma-3-27b-it": 60 / 10,        # 10 req/min = 6 sn bekle
        "glm-4-flash": 1.0,               # Güvenli aralık
        "deepseek-chat": 1.0,             # Güvenli aralık
    }
    
    def __init__(self, 
                 model_type: str,
                 model_name: str,
                 api_key: str):
        """
        LLM Evaluator'ı başlat
        
        Args:
            model_type: "gemini", "gemma", "glm", "deepseek", vb.
            model_name: "gemini-1.5-flash", "gemma-3-27b-it", vb.
            api_key: API anahtarınız
            
        Örnek:
            evaluator = LLMEvaluator("gemini", "gemini-1.5-flash", "AIzaSy...")
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.api_key = api_key
        
        # Rate limiting için bekleme süresi
        self.sleep_time = self.RATE_LIMITS.get(model_name, 1.0)
        
        # Model'i başlat
        self._initialize_model()
        
        print(f"✓ {model_type} modeli başlatıldı: {model_name}")
        print(f"  Rate limit: Her istek arası {self.sleep_time:.1f} saniye bekleme")
    
    def _initialize_model(self):
        if self.model_type in ["gemini", "gemma"]:
            if not GEMINI_AVAILABLE:
                raise ImportError("google-genai kurulu değil!")
            
            self.client = genai.Client(api_key=self.api_key)
            
            # Model adını düzelt (gemma için hâlâ prefix gerekebiliyor)
            if self.model_type == "gemma":
                self.model_name = f"models/{self.model_name}"   # ← burayı self.model_name olarak güncelle
            
          
                
        elif self.model_type in ["glm", "deepseek"]:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai kurulu değil!")
            
            # API base URL'leri
            base_urls = {
                "glm": "https://open.bigmodel.cn/api/paas/v4/",
                "deepseek": "https://api.deepseek.com"
            }
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_urls.get(self.model_type)
            )
        
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {self.model_type}")
            print("Desteklenen modeller: gemini, gemma, glm, deepseek")
    
    def evaluate_schema_pair(self, 
                           ground_truth: Dict,
                           prediction: Dict,
                           input_text: str) -> Dict[str, Any]:
        """
        İki şemayı karşılaştır ve değerlendir
        
        Args:
            ground_truth: Referans şema (schema_train.json'dan)
            prediction: Modelinizin ürettiği şema
            input_text: Orijinal radyoloji metni
            
        Returns:
            Değerlendirme sonuçları:
            {
                "similarity_score": 0.0-1.0,
                "clinical_equivalence": "exact|high|partial|low",
                "are_same_meaning": True/False,
                "entity_level_analysis": [...],
                "critical_errors": [...],
                "minor_differences": [...],
                "overall_assessment": "..."
            }
        """
        # Değerlendirme promptunu hazırla
        prompt = self._build_evaluation_prompt(ground_truth, prediction, input_text)
        
        # LLM'e sor
        response = self._generate(prompt)
        
        # Cevabı parse et
        result = self._parse_json_response(response)
        
        # Rate limit için bekle
        time.sleep(self.sleep_time)
        
        return result
    
    def batch_evaluate(self, 
                      comparisons: List[Dict],
                      save_every: int = 10) -> List[Dict]:
        """
        Birden fazla şema çiftini değerlendir
        
        Args:
            comparisons: [{"ground_truth": {...}, "prediction": {...}, "input": "..."}, ...]
            save_every: Her N değerlendirmede ara sonucu kaydet
            
        Returns:
            Tüm değerlendirme sonuçları listesi
        """
        results = []
        
        print(f"\n🔄 {len(comparisons)} şema çifti değerlendiriliyor...")
        print(f"⏱ Tahmini süre: {len(comparisons) * self.sleep_time / 60:.1f} dakika")
        
        for idx, comp in enumerate(comparisons):
            try:
                result = self.evaluate_schema_pair(
                    comp['ground_truth'],
                    comp['prediction'],
                    comp['input']
                )
                results.append(result)
                
                # İlerleme göster
                if (idx + 1) % 5 == 0:
                    print(f"  ✓ {idx + 1}/{len(comparisons)} tamamlandı")
                
                # Ara sonuçları kaydet
                if (idx + 1) % save_every == 0:
                    self._save_intermediate(results, idx + 1)
                
            except Exception as e:
                print(f"  ⚠ Hata (sample {idx}): {e}")
                results.append({
                    'error': str(e),
                    'sample_idx': idx
                })
        
        print(f"✓ Değerlendirme tamamlandı!")
        return results
    def _safe_extract_gemini_text(self, response) -> Optional[str]:
        """
        Gemini response'tan güvenli şekilde text çıkar
        (finish_reason=2 gibi durumları tolere eder)
        """
        try:
            if not hasattr(response, "candidates") or not response.candidates:
                return None

            candidate = response.candidates[0]

            if not hasattr(candidate, "content") or not candidate.content:
                return None

            parts = candidate.content.parts
            if not parts:
                return None

            texts = []
            for part in parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)

            return "\n".join(texts) if texts else None

        except Exception:
            return None


    def _generate(self, prompt: str) -> str:
        """
        Seçilen model ile text üret
        
        Her model tipi için farklı API çağrısı
        """
        try:
            if self.model_type in ["gemini", "gemma"]:
                return self._generate_gemini(prompt)
            
            elif self.model_type in ["glm", "deepseek"]:
                return self._generate_openai_compatible(prompt)
            
            else:
                raise ValueError(f"Model tipi desteklenmiyor: {self.model_type}")
        
        except Exception as e:
            logger.error(f"LLM generation hatası: {e}")
            raise
    
    def _generate_gemini(self, prompt: str) -> str:
        """Gemini/Gemma için üretim - google-genai SDK uyumlu"""

        # Safety ayarlarını tanımla
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
        ]

        # Config yapısı - Safety settings BURAYA eklenmeli
        config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.85,
            max_output_tokens=8192,
            safety_settings=safety_settings  # <--- Moved inside config
        )

        # API Çağrısı (safety_settings argümanı kaldırıldı)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=config
        )

        # Metni çek
        if response.text:
            return response.text.strip()

        # Alternatif (candidates yapısı)
        try:
            return response.candidates[0].content.parts[0].text.strip()
        except (AttributeError, IndexError, KeyError, TypeError):
            raise Exception(f"Gemini'den metin çıkarılamadı - response: {response}")


    
    def _generate_openai_compatible(self, prompt: str) -> str:
        """GLM, DeepSeek gibi OpenAI-compatible API'ler için üretim"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2048
        )
        
        return response.choices[0].message.content.strip()
    
    def _build_evaluation_prompt(self, 
                                 gt: Dict, 
                                 pred: Dict, 
                                 input_text: str) -> str:
        """
        Değerlendirme promptunu oluştur
        
        LLM'e hangi kriterlere göre değerlendireceğini söylüyoruz
        """
        
        # Ground truth ve prediction'ı JSON string'e çevir
        gt_output = json.dumps(gt.get('output', []), indent=2)
        pred_output = json.dumps(pred.get('output', []), indent=2)
        
        prompt = f"""You are an expert medical information extraction evaluator. Compare two medical schema extractions and determine if they are clinically equivalent.IMPORTANT:
- Ignore formatting differences (e.g., '13mm' vs '13 mm'). Focus only on clinical meaning conflicts.
- This is a text consistency evaluation task.
- Do NOT provide medical advice or diagnosis.
- Do NOT suggest treatment.


📋 INPUT RADIOLOGY TEXT:
{input_text}

📊 GROUND TRUTH SCHEMA (Reference):
{gt_output}

🤖 PREDICTED SCHEMA (System Output):
{pred_output}

📝 EVALUATION CRITERIA:

1. **Clinical Equivalence**
   - Do both schemas convey the same clinical information?
   - Consider: synonyms (e.g., "enlarged" = "splenomegaly")
   - Consider: paraphrasing (e.g., "no fracture" = "fracture absent")

2. **Entity Completeness**
   - Are all entities from ground truth captured?
   - Missing entities = critical error
   - Extra valid entities = acceptable

3. **Field Accuracy** (for each field):
   - abnormality: exact match best, semantic match acceptable
   - presence: CRITICAL - must match exactly (present/absent/uncertain)
   - location: consider anatomical synonyms
   - degree: consider equivalent descriptions
   - measurement: exact numbers important

4. **Clinical Impact**
   - Which differences are clinically significant?
   - Which are just formatting/wording differences?

Return this exact JSON structure (NO extra text, NO markdown):
{{
  "similarity_score": 0.0-1.0,
  "clinical_equivalence": "exact|high|partial|low",
  "are_same_meaning": true or false,
  "critical_errors": ["list if any"],
  "minor_differences": ["list if any"],
  "overall_assessment": "one sentence summary"
}}

IMPORTANT RULES:
- Be strict on presence/absence (critical for diagnosis)
- Be lenient on synonyms and paraphrasing
- Focus on clinical impact, not textual similarity
"""
        
        return prompt
    
    # llm_evaluator.py → _parse_json_response() düzelt

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """LLM cevabını parse et - robust version"""
        
        # Temizle
        response = response.strip()
        
        # Markdown kaldır
        if response.startswith('```json'):
            response = response.replace('```json', '', 1)
        if response.startswith('```'):
            response = response.replace('```', '', 1)
        if response.endswith('```'):
            response = response.rsplit('```', 1)[0]
        
        response = response.strip()
        
        # ⭐ FIX 1: Incomplete JSON repair
        # Eğer son } yoksa, ekle
        if response.count('{') > response.count('}'):
            response += '}' * (response.count('{') - response.count('}'))
        
        # ⭐ FIX 2: Unterminated strings
        # Son satırda tırnak açıksa kapat
        lines = response.split('\n')
        fixed_lines = []
        for line in lines:
            # Tek tırnak sayısı
            quote_count = line.count('"') - line.count('\\"')
            if quote_count % 2 == 1:  # Tek sayıda tırnak
                # Satır sonu virgül varsa kaldır, yoksa ekle
                if line.rstrip().endswith(','):
                    line = line.rstrip()[:-1] + '",'
                else:
                    line = line.rstrip() + '"'
            fixed_lines.append(line)
        
        response = '\n'.join(fixed_lines)
        
        # JSON parse
        try:
            parsed = json.loads(response)
            return parsed
        
        except json.JSONDecodeError as e:
            print(f"⚠ JSON parse hatası: {e}")
            print(f"Raw response (ilk 500 char): {response[:500]}")
            
            # ⭐ FIX 3: Fallback - minimal valid response
            try:
                # Extract similarity_score en azından
                import re
                sim_match = re.search(r'"similarity_score":\s*([0-9.]+)', response)
                sim_score = float(sim_match.group(1)) if sim_match else 0.0
                
                equiv_match = re.search(r'"clinical_equivalence":\s*"(\w+)"', response)
                equiv = equiv_match.group(1) if equiv_match else 'unknown'
                
                return {
                    'similarity_score': sim_score,
                    'clinical_equivalence': equiv,
                    'are_same_meaning': False,
                    'entity_level_analysis': [],
                    'critical_errors': ['Incomplete LLM response - partial data'],
                    'minor_differences': [],
                    'overall_assessment': 'Evaluation incomplete due to parse error',
                    'raw_response': response[:500],
                    'parse_error': str(e)
                }
            except:
                # Son çare - tamamen boş
                return {
                    'similarity_score': 0.0,
                    'clinical_equivalence': 'unknown',
                    'are_same_meaning': False,
                    'entity_level_analysis': [],
                    'critical_errors': ['Complete JSON parse failure'],
                    'minor_differences': [],
                    'overall_assessment': 'Evaluation failed',
                    'raw_response': response[:500],
                    'parse_error': str(e)
                }
    
    def _save_intermediate(self, results: List[Dict], count: int):
        """Ara sonuçları kaydet (hata durumunda kaybetmemek için)"""
        filename = f'llm_eval_intermediate_{count}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  💾 Ara sonuç kaydedildi: {filename}")


# ============================================================================
# Semantic Similarity (Embedding-based)
# ============================================================================

class EmbeddingBasedEvaluator:
    """
    Sentence embeddings kullanarak semantik benzerlik hesapla
    
    Bu method API çağrısı gerektirmez, tamamen local çalışır!
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Args:
            model_name: HuggingFace model adı
                       - 'sentence-transformers/all-MiniLM-L6-v2' (genel, hızlı)
                       - 'dmis-lab/biobert-base-cased-v1.2' (medikal, daha iyi)
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"✓ Embedding modeli yüklendi: {model_name}")
        except ImportError:
            print("⚠ sentence-transformers kurulu değil!")
            print("  pip install sentence-transformers")
            self.model = None
    
    def compute_schema_similarity(self, schema1: Dict, schema2: Dict) -> float:
        """
        İki şema arasında semantik benzerlik hesapla
        
        Returns:
            0.0 ile 1.0 arası similarity score
        """
        if not self.model:
            return 0.0
        
        # Şemaları text'e çevir
        text1 = self._schema_to_text(schema1)
        text2 = self._schema_to_text(schema2)
        
        # Embeddings hesapla
        embeddings = self.model.encode([text1, text2])
        
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    def _schema_to_text(self, schema: Dict) -> str:
        """
        Şemayı text'e çevir (embedding için)
        
        Tüm field'ları anlamlı bir text'e dönüştür
        """
        output = schema.get('output', [])
        
        texts = []
        for entity in output:
            parts = []
            
            # Her field'ı ekle
            if entity.get('abnormality') and entity['abnormality'] != 'None':
                parts.append(f"abnormality: {entity['abnormality']}")
            
            if entity.get('presence') and entity['presence'] != 'None':
                parts.append(f"presence: {entity['presence']}")
            
            if entity.get('location'):
                loc = entity['location']
                if isinstance(loc, list):
                    parts.append(f"location: {', '.join(loc)}")
                else:
                    parts.append(f"location: {loc}")
            
            if entity.get('degree') and entity['degree'] != 'None':
                deg = entity['degree']
                if isinstance(deg, list):
                    parts.append(f"degree: {', '.join(deg)}")
                else:
                    parts.append(f"degree: {deg}")
            
            if entity.get('measurement') and entity['measurement'] != 'None':
                parts.append(f"measurement: {entity['measurement']}")
            
            if parts:
                texts.append(' | '.join(parts))
        
        return ' ; '.join(texts)


# ============================================================================
# Kullanım Örnekleri
# ============================================================================

def example_basic_usage():
    """
    ÖRNEK 1: Temel kullanım (sizin kod yapınıza uygun)
    """
    print("\n" + "="*70)
    print("ÖRNEK 1: Temel LLM Değerlendirme")
    print("="*70)
    
    # API keylerini tanımla
    API_KEYS = {
        "gemini": "AIzaSyDKfk3iyWUilm8SU-f70PSRjo9etZBxrDk",
        "gemma": "KGAT_7b8482384bb20717b1fa8b9c914ff365",
        "glm": "sk-t80kLqA1bkLIoTi0x0vjmno3-gbMvrX3A44SOh4QWHRpiYJvMeOTpUOScAAWzOPzpDxC8AyC0KPdgaqHrn_5RPa_RhY_",
        "deepseek": "sk-450186e490b34beb8347badc0fa91e6b",
    }


    
    # Evaluator oluştur (model seçimi çok basit!)
    evaluator = LLMEvaluator(
        model_type="gemini",              # "gemini", "gemma", "glm", "deepseek"
        model_name="models/gemini-2.5-flash",    # Model adı
        api_key=API_KEYS["gemini"]        # API key
    )
    
    # Örnek şemalar
    ground_truth = {
        'input': 'Spleen size is enlarged and measured 134 mm.',
        'output': [{
            'abnormality': 'enlarged',
            'presence': 'present',
            'location': ['spleen'],
            'degree': 'None',
            'measurement': '134 mm',
            'comparison': 'None'
        }]
    }
    
    prediction = {
        'input': 'Spleen size is enlarged and measured 134 mm.',
        'output': [{
            'abnormality': 'splenomegaly',  # Farklı kelime, aynı anlam
            'presence': 'present',
            'location': ['spleen'],
            'degree': 'None',
            'measurement': '134mm',  # Boşluk farkı
            'comparison': 'None'
        }]
    }
    
    # Değerlendir!
    result = evaluator.evaluate_schema_pair(
        ground_truth,
        prediction,
        ground_truth['input']
    )
    
    # Sonuçları yazdır
    print(f"\n📊 SONUÇLAR:")
    print(f"  Benzerlik skoru: {result.get('similarity_score', 0):.3f}")
    print(f"  Klinik eşdeğerlik: {result.get('clinical_equivalence', 'unknown')}")
    print(f"  Aynı anlama mı geliyor: {result.get('are_same_meaning', False)}")
    print(f"\n  Değerlendirme: {result.get('overall_assessment', 'N/A')}")


def example_batch_evaluation():
    """
    ÖRNEK 2: Toplu değerlendirme (sizin map_schema yapınız gibi)
    """
    print("\n" + "="*70)
    print("ÖRNEK 2: Toplu Değerlendirme")
    print("="*70)
    
    # API key
    API_KEY = "AIzaSyDKfk3iyWUilm8SU-f70PSRjo9etZBxrDk"
    
    # Evaluator
    evaluator = LLMEvaluator(
        model_type="gemini",
        model_name="models/gemini-2.5-flash",
        api_key=API_KEY
    )
    
    # Ground truth ve predictions yükle
    with open('./data/0_normalized/gt0.json', 'r') as f:
        ground_truths = json.load(f)
    
    with open('./data/0_normalized/sample0.0.json', 'r') as f:
        predictions = json.load(f)
    
    # Karşılaştırma listesi hazırla
    comparisons = []
    for gt in ground_truths:
        # Matching prediction bul (input'a göre)
        pred = next(
            (p for p in predictions if p['input'] == gt['input']),
            None
        )
        
        if pred:
            comparisons.append({
                'ground_truth': gt,
                'prediction': pred,
                'input': gt['input']
            })
    
    # Toplu değerlendirme (sizin map_schema gibi)
    results = evaluator.batch_evaluate(comparisons)
    
    # Sonuçları kaydet
    with open('llm_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ {len(results)} değerlendirme tamamlandı!")


def example_multi_model_comparison():
    """
    ÖRNEK 3: Birden fazla model ile değerlendirme
    """
    print("\n" + "="*70)
    print("ÖRNEK 3: Multi-Model Karşılaştırma")
    print("="*70)
    
    # Tüm API keyler
    API_KEYS = {
        "gemini": "AIzaSyDKfk3iyWUilm8SU-f70PSRjo9etZBxrDk",
        "gemma": "KGAT_7b8482384bb20717b1fa8b9c914ff365",
    }
    
    # Örnek şema çifti
    gt = {...}
    pred = {...}
    
    # Her model ile değerlendir
    results = {}
    
    for model_type, api_key in API_KEYS.items():
        print(f"\n🔄 {model_type} ile değerlendiriliyor...")
        
        evaluator = LLMEvaluator(
            model_type=model_type,
            model_name=f"{model_type}-1.5-flash",
            api_key=api_key
        )
        
        result = evaluator.evaluate_schema_pair(gt, pred, gt['input'])
        results[model_type] = result
    
    # Karşılaştır
    print(f"\n📊 MODEL KARŞILAŞTIRMASI:")
    for model, result in results.items():
        print(f"  {model:10s}: similarity={result['similarity_score']:.3f}")


if __name__ == '__main__':
    
    # Örnekleri çalıştırmak isterseniz uncomment edin:
    # example_basic_usage()
    # example_batch_evaluation()
    # example_multi_model_comparison()
    pass