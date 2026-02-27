from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class BaseModalityWorkflow(ABC):
    """Tüm modaliteler için base class"""
    
    def __init__(self, modality: str):
        self.modality = modality
    
    @abstractmethod
    def pre_process(self, entities: List[Dict]) -> List[Dict]:
        """Entity'leri işlemeden önce modaliteye özel düzeltmeler"""
        pass
    
    @abstractmethod
    def post_process(self, entities: List[Dict], violations: List) -> Tuple[List[Dict], Dict]:
        """İşlemden sonra modaliteye özel kontroller ve istatistikler"""
        pass
    
    def validate_modality(self, report_text: str) -> bool:
        """Rapor metninin bu modaliteye ait olduğunu doğrula"""
        # Simple keyword matching (ileride ML-based yapılabilir)
        keywords = self._get_modality_keywords()
        text_lower = report_text.lower()
        return any(kw in text_lower for kw in keywords)
    
    def _get_modality_keywords(self) -> List[str]:
        """Modalite anahtar kelimeleri"""
        if self.modality == "chest_ct":
            return ['chest', 'thorax', 'thoracic', 'lung', 'ct chest', 'chest ct']
        return []