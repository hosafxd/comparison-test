# ============================================================================
# multi_embedding_evaluator.py - EMBEDDING EVALUATOR
# ============================================================================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EmbeddingEvaluator:
    """
    Tek bir embedding model ile semantic similarity hesaplama
    """
    
    def __init__(self, model_name):
        """
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        print(f"    Loading: {model_name}...")
        
        try:
            self.model = SentenceTransformer(model_name)
            print(f"    ✅ Loaded successfully")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            self.model = None
    
  

    def schema_to_text(self, schema):
        """Convert schema to text with weighted importance"""
        output = schema.get('output', [])
        
        texts = []
        for entity in output:
            parts = []
            
            # ⭐ PRESENCE ÇOK ÖNEMLİ - 3x tekrarla
            presence = entity.get('presence')
            if presence and presence not in ['None', 'unknown']:
                # Presence'ı vurgula
                parts.append(f"PRESENCE: {presence}")
                parts.append(f"{presence}")  # Tekrar
                parts.append(f"status {presence}")  # Tekrar
            
            # Abnormality (2x)
            abnormality = entity.get('abnormality')
            if abnormality and abnormality != 'None':
                parts.append(f"abnormality {abnormality}")
                parts.append(f"{abnormality}")
            
            # Location (2x)
            location = entity.get('location')
            if location and location != ['None']:
                if isinstance(location, list):
                    loc_str = ' '.join(str(v) for v in location)
                else:
                    loc_str = str(location)
                parts.append(f"location {loc_str}")
                parts.append(f"in {loc_str}")
            
            # Degree (1x)
            degree = entity.get('degree')
            if degree and degree != ['None']:
                if isinstance(degree, list):
                    parts.append(' '.join(str(v) for v in degree))
                else:
                    parts.append(str(degree))
            
            # Measurement (1x)
            measurement = entity.get('measurement')
            if measurement and measurement != 'None':
                parts.append(f"{measurement}")
            
            # Finding (optional, 0.5x)
            finding = entity.get('finding')
            if finding and finding != 'None' and len(parts) < 10:
                parts.append(f"{finding}")
            
            if parts:
                texts.append(' '.join(parts))
        
        return '. '.join(texts) + '.'  # Sentence sonlandır
    
    def compute_similarity(self, schema1, schema2):
        """Compute cosine similarity between two schemas"""
        if not self.model:
            return 0.0
        
        text1 = self.schema_to_text(schema1)
        text2 = self.schema_to_text(schema2)
        
        embeddings = self.model.encode([text1, text2])
        
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        

        return float(similarity)