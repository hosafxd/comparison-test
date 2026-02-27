"""
MODALITY ROUTER
Raporu analiz eder, modaliteyi tespit eder ve ilgili workflow'a yönlendirir
"""

import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Workflow imports (absolute)
try:
    from workflows.chest_ct_workflow import ChestCTWorkflow
except ImportError:
    from chest_ct_workflow import ChestCTWorkflow

# İleride eklenecekler:
# from workflows.brain_ct_workflow import BrainCTWorkflow


class ModalityRouter:
    """
    Rapor metninden modaliteyi tespit eder ve uygun workflow'u seçer
    """
    
    SUPPORTED_MODALITIES = {
        'chest_ct': ChestCTWorkflow,
        # 'brain_ct': BrainCTWorkflow,
    }
    
    def __init__(self):
        self.workflows = {}
        self._load_workflows()
    
    def _load_workflows(self):
        """Workflow instance'larını önbelleğe al"""
        for modality, workflow_class in self.SUPPORTED_MODALITIES.items():
            self.workflows[modality] = workflow_class()
    
    def detect_modality(self, report_text: str, schema: Dict = None) -> str:
        text_lower = report_text.lower()
        
        # === CHEST CT DETECTION ===
        # Tier 1: Direct modality keywords (high confidence)
        chest_ct_keywords = [
            'chest ct', 'ct chest', 'thoracic ct', 'ct thorax',
            'computed tomography of the chest', 'chest computed tomography',
            'ct of the chest', 'ct scan of chest', 'cta chest',
            'ct pulmonary angiography', 'ctpa', 'hrct chest',
            'non-contrast ct chest', 'contrast-enhanced ct chest',
            'ct scan of the thorax',
        ]

        if any(kw in text_lower for kw in chest_ct_keywords):
            return 'chest_ct'

        # Fallback: anatomik keyword inference
        # Raporda 3+ thoracic anatomy terimi varsa → chest_ct
        thoracic_terms = ['pulmonary', 'cardiac', 'mediastinal', 'pleural',
                        'pericardial', 'aortic', 'lung', 'bronch', 'trachea',
                        'thoracic', 'hilar', 'lobar']
        thoracic_count = sum(1 for t in thoracic_terms if t in text_lower)
        if thoracic_count >= 3:
            return 'chest_ct'
        
        # Tier 3: Schema-based inference (if schema metadata available)
        if schema:
            modality_field = schema.get('modality', '').lower()
            if 'ct' in modality_field and any(t in modality_field for t in ['chest', 'thorax', 'thoracic']):
                return 'chest_ct'
        
        # === BRAIN CT DETECTION ===
        brain_keywords = ['brain ct', 'ct brain', 'head ct', 'ct head', 
                         'cranial ct', 'ct of the brain', 'ct of the head',
                         'ct scan of the head', 'non-contrast ct head']
        if any(kw in text_lower for kw in brain_keywords):
            return 'brain_ct'
        
        return 'general'
    
    def get_workflow(self, modality: str):
        """Modaliteye göre workflow döndür"""
        if modality in self.workflows:
            return self.workflows[modality]
        else:
            print(f"⚠️  Modality '{modality}' not supported, using general evaluation")
            return None