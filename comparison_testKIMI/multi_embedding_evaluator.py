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
        """Convert schema with AGGRESSIVE weighting"""
        output = schema.get('output', [])
        
        texts = []
        for entity in output:
            parts = []
            
            # ⭐ PRESENCE 5x (en kritik)
            presence = entity.get('finding_presence') or entity.get('presence')
            if presence and presence not in ['None', 'unknown']:
                for _ in range(5):  # 5x repeat
                    parts.append(f"STATUS_{presence.upper()}")
            
            # General finding 3x
            gen_finding = entity.get('general_finding')
            if gen_finding and gen_finding != 'None':
                for _ in range(3):
                    parts.append(gen_finding)
            
            # Specific finding 2x
            spec_finding = entity.get('specific_finding')
            if spec_finding and spec_finding != 'None':
                for _ in range(2):
                    parts.append(spec_finding)
            
            # Location 2x
            location = entity.get('location', [])
            if location and location != ['None']:
                if isinstance(location, list):
                    loc_str = ' '.join(str(v) for v in location)
                else:
                    loc_str = str(location)
                for _ in range(2):
                    parts.append(f"LOC_{loc_str}")
            
            # Degree, measurement (1x each)
            for field in ['degree', 'measurement']:
                value = entity.get(field)
                if value and value != 'None':
                    if isinstance(value, list):
                        parts.append(' '.join(str(v) for v in value))
                    else:
                        parts.append(str(value))
            
            if parts:
                texts.append(' '.join(parts))
        
        return '. '.join(texts) + '.'
    
    def compute_similarity(self, schema1, schema2):
        """Compute cosine similarity between two schemas"""
        if not self.model:
            return 0.0
        
        text1 = self.schema_to_text(schema1)
        text2 = self.schema_to_text(schema2)
        
        embeddings = self.model.encode([text1, text2])
        
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        

        return float(similarity)