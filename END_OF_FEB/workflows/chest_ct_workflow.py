"""
THORACIC IMAGING ONTOLOGY
Entity'lerin hiyerarşik sınıflandırması ve semantic ilişkileri
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
try:
    from .base_workflow import BaseModalityWorkflow  # Relative import (paket olarak kullanım)
except ImportError:
    from base_workflow import BaseModalityWorkflow   # Direct run için

class FindingCategory(Enum):
    PULMONARY_VASCULAR = "pulmonary_vascular_disorders"
    AORTIC_ARTERIAL = "aortic_and_major_arterial_pathology"
    CORONARY = "coronary_artery_findings"
    CARDIAC_STRUCTURAL = "cardiac_structural_functional"
    PERICARDIAL = "pericardial_disorders"
    VALVULAR = "valvular_disorders"
    PULMONARY_INFECTION = "pulmonary_infections_inflammatory"
    INTERSTITIAL_LUNG = "interstitial_lung_disease"
    PARENCHYMAL = "parenchymal_opacities"
    AIRWAY = "airway_disorders"
    ATELECTASIS = "atelectasis_collapse"
    PLEURAL = "pleural_disorders"
    THORACIC_MASSES = "thoracic_masses_nodules"
    ESOPHAGEAL_DIAPHRAGMATIC = "esophageal_diaphragmatic"


@dataclass
class ClinicalEntity:
    canonical_name: str
    synonyms: Set[str]
    category: FindingCategory
    parent_concepts: Set[str]
    subtypes: Set[str] = field(default_factory=set)
    severity_indicators: Set[str] = field(default_factory=set)
    typical_locations: Set[str] = field(default_factory=set)
    conflicts_with: Set[str] = field(default_factory=set)


class ThoracicOntology:
    """
    Göğüs görüntüleme için klinik ontoloji.
    Entity normalization ve hierarchical matching için kullanılır.
    """

    def __init__(self):
        self.entity_map: Dict[str, ClinicalEntity] = {}          # canonical_name -> entity
        self.synonym_index: Dict[str, str] = {}                   # lowercase synonym -> canonical_name
        self.category_map: Dict[FindingCategory, Set[str]] = {
            cat: set() for cat in FindingCategory
        }
        self._build_ontology()

    # ------------------------------------------------------------------ #
    #  INTERNAL HELPERS
    # ------------------------------------------------------------------ #
    def _add_entity(self, entity: ClinicalEntity) -> None:
        """Ontolojiye yeni bir entity ekle ve index'leri güncelle."""
        canon = entity.canonical_name
        self.entity_map[canon] = entity
        self.category_map[entity.category].add(canon)

        # Synonym index (case-insensitive)
        self.synonym_index[canon.lower()] = canon
        for syn in entity.synonyms:
            self.synonym_index[syn.lower()] = canon

    # ------------------------------------------------------------------ #
    #  PUBLIC LOOKUP API
    # ------------------------------------------------------------------ #
    def resolve(self, term: str) -> Optional[ClinicalEntity]:
        """
        Verilen terimi (synonym dahil) canonical entity'ye çözümle.
        Bulunamazsa None döner.
        """
        canon = self.synonym_index.get(term.lower())
        if canon:
            return self.entity_map[canon]
        return None

    def get_canonical_name(self, term: str) -> Optional[str]:
        """Verilen synonym/alias'tan canonical name döndür."""
        return self.synonym_index.get(term.lower())

    def get_category_entities(self, category: FindingCategory) -> List[ClinicalEntity]:
        """Belirli bir kategorideki tüm entity'leri döndür."""
        return [self.entity_map[name] for name in sorted(self.category_map[category])]

    def are_conflicting(self, term_a: str, term_b: str) -> bool:
        """İki bulgu birbiriyle çelişiyor mu? (synonym-aware)"""
        entity_a = self.resolve(term_a)
        entity_b = self.resolve(term_b)
        if not entity_a or not entity_b:
            return False

        # b'nin canonical veya herhangi bir synonym'i a'nın conflicts setinde mi?
        b_names = {entity_b.canonical_name} | entity_b.synonyms
        a_names = {entity_a.canonical_name} | entity_a.synonyms

        if b_names & entity_a.conflicts_with:
            return True
        if a_names & entity_b.conflicts_with:
            return True
        return False

    def is_subtype_of(self, child_term: str, parent_term: str) -> bool:
        """child_term, parent_term'in alt tipi mi?"""
        parent = self.resolve(parent_term)
        if not parent:
            return False
        child_canon = self.get_canonical_name(child_term)
        return child_canon is not None and child_canon in parent.subtypes

    def fuzzy_search(self, query: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Basit token-overlap tabanlı fuzzy arama.
        (query, score) listesi döndürür – score azalan sırada.
        """
        query_tokens = set(query.lower().split())
        results: List[Tuple[str, float]] = []
        for key in self.synonym_index:
            key_tokens = set(key.split())
            if not key_tokens:
                continue
            overlap = len(query_tokens & key_tokens)
            score = overlap / max(len(query_tokens), len(key_tokens))
            if score >= threshold:
                canon = self.synonym_index[key]
                results.append((canon, score))
        # Deduplicate by canonical name, keep best score
        best: Dict[str, float] = {}
        for canon, score in results:
            if canon not in best or score > best[canon]:
                best[canon] = score
        return sorted(best.items(), key=lambda x: x[1], reverse=True)

    def summary(self) -> Dict[str, int]:
        """Kategori başına entity sayısı."""
        return {cat.value: len(names) for cat, names in self.category_map.items()}

    # ------------------------------------------------------------------ #
    #  ONTOLOGY BUILDER
    # ------------------------------------------------------------------ #
    def _build_ontology(self):
        """Thoracic imaging entity'lerini tanımla."""

        # =============================================================
        # 1. PULMONARY VASCULAR DISORDERS
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Pulmonary Embolism",
            synonyms={"PE", "Pulmonary Thromboembolism", "Pulmonary Embolus", "Lung Clot"},
            category=FindingCategory.PULMONARY_VASCULAR,
            parent_concepts={"Pulmonary Vascular Disorder", "Vascular Emergency", "Thromboembolic Disease"},
            subtypes={"Chronic Pulmonary Embolism", "Acute Pulmonary Embolism", "Submassive PE", "Massive PE"},
            severity_indicators={"massive", "submassive", "chronic", "acute"},
            typical_locations={"pulmonary artery", "main pulmonary artery", "lobar pulmonary artery",
                               "segmental pulmonary artery"},
            conflicts_with={"Normal Pulmonary Vasculature"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Pulmonary Infarction",
            synonyms={"Lung Infarct", "Pulmonary Infarct", "Hampton Hump"},
            category=FindingCategory.PULMONARY_VASCULAR,
            parent_concepts={"Pulmonary Vascular Disorder", "Thromboembolic Disease"},
            subtypes=set(),
            severity_indicators={"acute", "subacute"},
            typical_locations={"lower lobes", "peripheral lung", "subpleural"},
            conflicts_with={"Normal Pulmonary Parenchyma"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Pulmonary Hypertension",
            synonyms={"PAH", "Pulmonary Arterial Hypertension", "Elevated Pulmonary Pressure"},
            category=FindingCategory.PULMONARY_VASCULAR,
            parent_concepts={"Pulmonary Vascular Disorder"},
            subtypes={"Primary Pulmonary Hypertension", "Secondary Pulmonary Hypertension"},
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"main pulmonary artery", "pulmonary arteries"},
            conflicts_with={"Normal Pulmonary Artery Caliber"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Pulmonary Artery Aneurysm",
            synonyms={"PA Aneurysm", "Pulmonary Arterial Aneurysm", "Pulmonary Artery Dilatation"},
            category=FindingCategory.PULMONARY_VASCULAR,
            parent_concepts={"Pulmonary Vascular Disorder", "Vascular Aneurysm"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"main pulmonary artery", "right pulmonary artery", "left pulmonary artery"},
            conflicts_with={"Normal Pulmonary Artery Caliber"},
        ))

        # =============================================================
        # 2. AORTIC AND MAJOR ARTERIAL PATHOLOGY
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Aortic Dissection",
            synonyms={"Dissection", "Aortic Tear", "Intimal Flap", "Aortic Rupture Risk"},
            category=FindingCategory.AORTIC_ARTERIAL,
            parent_concepts={"Aortic Pathology", "Vascular Emergency", "Acute Aortic Syndrome"},
            subtypes={"Type A Dissection", "Type B Dissection", "Stanford A", "Stanford B",
                       "DeBakey I", "DeBakey II", "DeBakey III"},
            severity_indicators={"acute", "chronic", "complicated", "uncomplicated"},
            typical_locations={"ascending aorta", "aortic arch", "descending aorta", "aortic root"},
            conflicts_with={"Normal Aorta"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Aortic Intramural Hematoma",
            synonyms={"IMH", "Intramural Hematoma", "Aortic Wall Hematoma",
                       "Dissection without Intimal Tear"},
            category=FindingCategory.AORTIC_ARTERIAL,
            parent_concepts={"Aortic Pathology", "Acute Aortic Syndrome"},
            subtypes=set(),
            severity_indicators={"acute", "subacute", "chronic"},
            typical_locations={"descending aorta", "ascending aorta"},
            conflicts_with={"Normal Aortic Wall"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Penetrating Atherosclerotic Ulcer",
            synonyms={"PAU", "Aortic Ulcer", "Penetrating Ulcer", "Atherosclerotic Ulcer"},
            category=FindingCategory.AORTIC_ARTERIAL,
            parent_concepts={"Aortic Pathology", "Acute Aortic Syndrome"},
            subtypes=set(),
            severity_indicators={"complicated", "uncomplicated"},
            typical_locations={"descending aorta", "thoracic aorta"},
            conflicts_with={"Normal Aortic Wall"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Thoracic Aortic Aneurysm",
            synonyms={"TAA", "Thoracic Aneurysm", "Aortic Dilatation"},
            category=FindingCategory.AORTIC_ARTERIAL,
            parent_concepts={"Aortic Pathology", "Vascular Aneurysm"},
            subtypes={"Ascending Aortic Aneurysm", "Aortic Arch Aneurysm", "Descending Aortic Aneurysm"},
            severity_indicators={"mild", "moderate", "severe", "ruptured"},
            typical_locations={"ascending aorta", "aortic arch", "descending aorta"},
            conflicts_with={"Normal Aortic Diameter"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Abdominal Aortic Aneurysm",
            synonyms={"AAA", "Abdominal Aneurysm"},
            category=FindingCategory.AORTIC_ARTERIAL,
            parent_concepts={"Aortic Pathology", "Vascular Aneurysm"},
            subtypes={"Infrarenal AAA", "Suprarenal AAA", "Juxtarenal AAA"},
            severity_indicators={"mild", "moderate", "severe", "ruptured"},
            typical_locations={"abdominal aorta", "infrarenal aorta"},
            conflicts_with={"Normal Abdominal Aorta"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Aortic Rupture",
            synonyms={"Ruptured Aorta", "Aortic Transection", "Traumatic Aortic Injury"},
            category=FindingCategory.AORTIC_ARTERIAL,
            parent_concepts={"Aortic Pathology", "Vascular Emergency"},
            subtypes={"Contained Rupture", "Free Rupture"},
            severity_indicators={"contained", "free", "acute"},
            typical_locations={"aortic isthmus", "ascending aorta", "descending aorta"},
            conflicts_with={"Normal Aorta"},
        ))

        # =============================================================
        # 3. CORONARY ARTERY FINDINGS
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Coronary Artery Calcification",
            synonyms={"CAC", "Coronary Calcification", "Coronary Calcium",
                       "Coronary Artery Calcium Score"},
            category=FindingCategory.CORONARY,
            parent_concepts={"Coronary Artery Disease", "Atherosclerosis"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe", "extensive"},
            typical_locations={"LAD", "left anterior descending", "RCA", "right coronary artery",
                               "LCx", "left circumflex", "left main"},
            conflicts_with={"Normal Coronary Arteries"},
        ))

        # =============================================================
        # 4. CARDIAC STRUCTURAL AND FUNCTIONAL DISORDERS
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Cardiomegaly",
            synonyms={"Enlarged Heart", "Cardiac Enlargement", "Increased Cardiac Silhouette"},
            category=FindingCategory.CARDIAC_STRUCTURAL,
            parent_concepts={"Cardiac Structural Disorder"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"heart"},
            conflicts_with={"Normal Cardiac Size"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Myocarditis",
            synonyms={"Myocardial Inflammation", "Inflammatory Cardiomyopathy"},
            category=FindingCategory.CARDIAC_STRUCTURAL,
            parent_concepts={"Cardiac Structural Disorder", "Inflammatory Cardiac Disease"},
            subtypes={"Acute Myocarditis", "Chronic Myocarditis", "Giant Cell Myocarditis"},
            severity_indicators={"acute", "chronic", "fulminant"},
            typical_locations={"myocardium", "left ventricle", "right ventricle"},
            conflicts_with={"Normal Myocardium"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Dilated Cardiomyopathy",
            synonyms={"DCM", "Dilated CM", "Congestive Cardiomyopathy"},
            category=FindingCategory.CARDIAC_STRUCTURAL,
            parent_concepts={"Cardiomyopathy", "Cardiac Structural Disorder"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"left ventricle", "biventricular"},
            conflicts_with={"Normal Ventricular Size", "Hypertrophic Cardiomyopathy"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Hypertrophic Cardiomyopathy",
            synonyms={"HCM", "HOCM", "Hypertrophic Obstructive Cardiomyopathy",
                       "Asymmetric Septal Hypertrophy"},
            category=FindingCategory.CARDIAC_STRUCTURAL,
            parent_concepts={"Cardiomyopathy", "Cardiac Structural Disorder"},
            subtypes={"Obstructive HCM", "Non-obstructive HCM", "Apical HCM"},
            severity_indicators={"mild", "moderate", "severe", "obstructive"},
            typical_locations={"interventricular septum", "left ventricle", "apex"},
            conflicts_with={"Normal Myocardial Thickness", "Dilated Cardiomyopathy"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Left Ventricular Hypertrophy",
            synonyms={"LVH", "LV Hypertrophy", "Left Ventricular Enlargement"},
            category=FindingCategory.CARDIAC_STRUCTURAL,
            parent_concepts={"Cardiac Structural Disorder", "Ventricular Hypertrophy"},
            subtypes={"Concentric LVH", "Eccentric LVH"},
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"left ventricle", "interventricular septum"},
            conflicts_with={"Normal Left Ventricular Wall Thickness"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Right Ventricular Enlargement",
            synonyms={"RV Enlargement", "RV Dilatation", "Right Ventricular Dilatation",
                       "Cor Pulmonale"},
            category=FindingCategory.CARDIAC_STRUCTURAL,
            parent_concepts={"Cardiac Structural Disorder"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"right ventricle"},
            conflicts_with={"Normal Right Ventricular Size"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Atrial Enlargement",
            synonyms={"Atrial Dilatation", "Enlarged Atrium"},
            category=FindingCategory.CARDIAC_STRUCTURAL,
            parent_concepts={"Cardiac Structural Disorder"},
            subtypes={"Left Atrial Enlargement", "Right Atrial Enlargement", "Biatrial Enlargement"},
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"left atrium", "right atrium"},
            conflicts_with={"Normal Atrial Size"},
        ))

        # =============================================================
        # 5. PERICARDIAL DISORDERS
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Pericardial Effusion",
            synonyms={"Pericardial Fluid", "Fluid Around Heart"},
            category=FindingCategory.PERICARDIAL,
            parent_concepts={"Pericardial Disorder"},
            subtypes={"Hemorrhagic Pericardial Effusion", "Serous Pericardial Effusion"},
            severity_indicators={"trivial", "small", "moderate", "large"},
            typical_locations={"pericardial space", "pericardium"},
            conflicts_with={"Normal Pericardium"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Pericardial Tamponade",
            synonyms={"Cardiac Tamponade", "Tamponade", "Tamponade Physiology"},
            category=FindingCategory.PERICARDIAL,
            parent_concepts={"Pericardial Disorder", "Cardiac Emergency"},
            subtypes=set(),
            severity_indicators={"acute", "subacute"},
            typical_locations={"pericardial space"},
            conflicts_with={"Normal Pericardium"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Constrictive Pericarditis",
            synonyms={"Pericardial Constriction", "Constrictive Pericardial Disease",
                       "Pericardial Calcification"},
            category=FindingCategory.PERICARDIAL,
            parent_concepts={"Pericardial Disorder", "Chronic Pericardial Disease"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"pericardium"},
            conflicts_with={"Normal Pericardium"},
        ))

        # =============================================================
        # 6. VALVULAR DISORDERS
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Valvular Stenosis",
            synonyms={"Valve Stenosis", "Stenotic Valve"},
            category=FindingCategory.VALVULAR,
            parent_concepts={"Valvular Heart Disease"},
            subtypes={"Aortic Stenosis", "Mitral Stenosis", "Pulmonic Stenosis",
                       "Tricuspid Stenosis"},
            severity_indicators={"mild", "moderate", "severe", "critical"},
            typical_locations={"aortic valve", "mitral valve", "pulmonic valve", "tricuspid valve"},
            conflicts_with={"Normal Valve Function"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Valvular Regurgitation",
            synonyms={"Valve Regurgitation", "Valvular Insufficiency", "Valve Incompetence",
                       "Valve Leak"},
            category=FindingCategory.VALVULAR,
            parent_concepts={"Valvular Heart Disease"},
            subtypes={"Aortic Regurgitation", "Mitral Regurgitation", "Pulmonic Regurgitation",
                       "Tricuspid Regurgitation"},
            severity_indicators={"trivial", "mild", "moderate", "severe"},
            typical_locations={"aortic valve", "mitral valve", "pulmonic valve", "tricuspid valve"},
            conflicts_with={"Normal Valve Function"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Mitral Annular Calcification",
            synonyms={"MAC", "Mitral Calcification", "Mitral Annulus Calcification"},
            category=FindingCategory.VALVULAR,
            parent_concepts={"Valvular Heart Disease", "Degenerative Valve Disease"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"mitral annulus"},
            conflicts_with={"Normal Mitral Annulus"},
        ))

        # =============================================================
        # 7. PULMONARY INFECTIONS AND INFLAMMATORY DISEASES
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Pneumonia",
            synonyms={"Lung Infection", "Pulmonary Infection", "Infectious Pneumonia"},
            category=FindingCategory.PULMONARY_INFECTION,
            parent_concepts={"Pulmonary Infection", "Inflammatory Lung Disease"},
            subtypes={"Bacterial Pneumonia", "Viral Pneumonia", "Fungal Pneumonia",
                       "Lobar Pneumonia", "Bronchopneumonia"},
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"right lower lobe", "left lower lobe", "right middle lobe",
                               "bilateral", "multilobar"},
            conflicts_with={"Normal Lung Parenchyma"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Aspiration Pneumonia",
            synonyms={"Aspiration Pneumonitis", "Aspiration", "Aspiration Lung Injury"},
            category=FindingCategory.PULMONARY_INFECTION,
            parent_concepts={"Pulmonary Infection", "Aspiration Event"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"right lower lobe", "posterior segments", "dependent portions"},
            conflicts_with={"Normal Lung Parenchyma"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Organizing Pneumonia",
            synonyms={"OP", "COP", "Cryptogenic Organizing Pneumonia", "BOOP",
                       "Bronchiolitis Obliterans Organizing Pneumonia"},
            category=FindingCategory.PULMONARY_INFECTION,
            parent_concepts={"Inflammatory Lung Disease"},
            subtypes={"Cryptogenic OP", "Secondary OP"},
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"peripheral", "subpleural", "peribronchial", "bilateral"},
            conflicts_with={"Normal Lung Parenchyma"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Pulmonary Fibrosis",
            synonyms={"Lung Fibrosis", "Fibrotic Lung Disease", "IPF",
                       "Idiopathic Pulmonary Fibrosis"},
            category=FindingCategory.PULMONARY_INFECTION,
            parent_concepts={"Interstitial Lung Disease", "Chronic Lung Disease"},
            subtypes={"Idiopathic Pulmonary Fibrosis", "Secondary Pulmonary Fibrosis"},
            severity_indicators={"mild", "moderate", "severe", "end-stage"},
            typical_locations={"bilateral lower lobes", "peripheral", "subpleural", "basal"},
            conflicts_with={"Normal Lung Parenchyma"},
        ))

        # =============================================================
        # 8. INTERSTITIAL LUNG DISEASE PATTERNS
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Usual Interstitial Pneumonia Pattern",
            synonyms={"UIP", "UIP Pattern", "Usual Interstitial Pneumonia",
                       "Definite UIP Pattern"},
            category=FindingCategory.INTERSTITIAL_LUNG,
            parent_concepts={"Interstitial Lung Disease", "ILD Pattern"},
            subtypes={"Definite UIP", "Probable UIP", "Indeterminate for UIP"},
            severity_indicators={"mild", "moderate", "severe", "end-stage"},
            typical_locations={"bilateral lower lobes", "subpleural", "basal predominant"},
            conflicts_with={"Nonspecific Interstitial Pneumonia Pattern", "Normal Lung Parenchyma"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Nonspecific Interstitial Pneumonia Pattern",
            synonyms={"NSIP", "NSIP Pattern", "Nonspecific Interstitial Pneumonia"},
            category=FindingCategory.INTERSTITIAL_LUNG,
            parent_concepts={"Interstitial Lung Disease", "ILD Pattern"},
            subtypes={"Cellular NSIP", "Fibrotic NSIP"},
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"bilateral lower lobes", "subpleural sparing", "peribronchial"},
            conflicts_with={"Usual Interstitial Pneumonia Pattern", "Normal Lung Parenchyma"},
        ))

        # =============================================================
        # 9. PARENCHYMAL OPACITIES AND IMAGING SIGNS
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Ground Glass Opacity",
            synonyms={"GGO", "Ground Glass", "Ground Glass Attenuation",
                       "Ground Glass Opacification"},
            category=FindingCategory.PARENCHYMAL,
            parent_concepts={"Parenchymal Opacity", "Imaging Sign"},
            subtypes={"Focal GGO", "Diffuse GGO", "Multifocal GGO"},
            severity_indicators={"focal", "multifocal", "diffuse"},
            typical_locations={"bilateral", "peripheral", "central", "upper lobes", "lower lobes"},
            conflicts_with={"Normal Lung Attenuation"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Consolidation",
            synonyms={"Pulmonary Consolidation", "Lung Consolidation", "Airspace Consolidation",
                       "Airspace Opacity"},
            category=FindingCategory.PARENCHYMAL,
            parent_concepts={"Parenchymal Opacity", "Imaging Sign"},
            subtypes={"Lobar Consolidation", "Segmental Consolidation", "Patchy Consolidation"},
            severity_indicators={"focal", "multifocal", "diffuse"},
            typical_locations={"right lower lobe", "left lower lobe", "bilateral", "multilobar"},
            conflicts_with={"Normal Lung Parenchyma"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Tree-in-Bud Nodularity",
            synonyms={"Tree in Bud", "TIB", "Tree-in-Bud Pattern", "Tree in Bud Sign",
                       "Endobronchial Spread"},
            category=FindingCategory.PARENCHYMAL,
            parent_concepts={"Parenchymal Opacity", "Small Airway Disease", "Imaging Sign"},
            subtypes=set(),
            severity_indicators={"focal", "multifocal", "diffuse"},
            typical_locations={"peripheral", "centrilobular", "bilateral"},
            conflicts_with={"Normal Lung Parenchyma"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Honeycombing",
            synonyms={"Honeycomb Pattern", "Honeycomb Lung", "Honeycomb Change"},
            category=FindingCategory.PARENCHYMAL,
            parent_concepts={"Parenchymal Opacity", "Fibrotic Change", "Imaging Sign"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"subpleural", "bilateral lower lobes", "basal"},
            conflicts_with={"Normal Lung Parenchyma"},
        ))

        # =============================================================
        # 10. AIRWAY DISORDERS
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Bronchiectasis",
            synonyms={"Bronchial Dilatation", "Dilated Bronchi", "Airway Dilatation"},
            category=FindingCategory.AIRWAY,
            parent_concepts={"Airway Disorder", "Chronic Airway Disease"},
            subtypes={"Cylindrical Bronchiectasis", "Varicose Bronchiectasis",
                       "Cystic Bronchiectasis", "Traction Bronchiectasis"},
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"bilateral lower lobes", "right middle lobe", "lingula"},
            conflicts_with={"Normal Bronchial Caliber"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Bronchiolectasis",
            synonyms={"Bronchiolar Dilatation", "Dilated Bronchioles"},
            category=FindingCategory.AIRWAY,
            parent_concepts={"Airway Disorder", "Small Airway Disease"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"peripheral lung", "bilateral"},
            conflicts_with={"Normal Bronchiolar Caliber"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Mucus Plugging",
            synonyms={"Mucoid Impaction", "Mucus Impaction", "Endobronchial Mucus",
                       "Finger in Glove Sign"},
            category=FindingCategory.AIRWAY,
            parent_concepts={"Airway Disorder", "Airway Obstruction"},
            subtypes=set(),
            severity_indicators={"focal", "diffuse"},
            typical_locations={"bronchi", "segmental bronchi", "central airways"},
            conflicts_with={"Patent Airways"},
        ))

        # =============================================================
        # 11. ATELECTASIS AND LUNG COLLAPSE
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Atelectasis",
            synonyms={"Lung Collapse", "Collapsed Lung", "Pulmonary Atelectasis"},
            category=FindingCategory.ATELECTASIS,
            parent_concepts={"Volume Loss", "Lung Collapse"},
            subtypes={"Compressive Atelectasis", "Obstructive Atelectasis",
                       "Passive Atelectasis", "Cicatricial Atelectasis",
                       "Rounded Atelectasis", "Plate Atelectasis"},
            severity_indicators={"mild", "moderate", "severe", "complete"},
            typical_locations={"left lower lobe", "right lower lobe", "bilateral bases",
                               "right middle lobe"},
            conflicts_with={"Normal Lung Volume"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Subsegmental Atelectasis",
            synonyms={"Discoid Atelectasis", "Plate-like Atelectasis", "Linear Atelectasis",
                       "Band Atelectasis"},
            category=FindingCategory.ATELECTASIS,
            parent_concepts={"Atelectasis", "Volume Loss"},
            subtypes=set(),
            severity_indicators={"minimal", "mild"},
            typical_locations={"bilateral bases", "lower lobes"},
            conflicts_with={"Normal Lung Volume"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Lobar Collapse",
            synonyms={"Lobar Atelectasis", "Complete Lobar Collapse"},
            category=FindingCategory.ATELECTASIS,
            parent_concepts={"Atelectasis", "Volume Loss"},
            subtypes={"Right Upper Lobe Collapse", "Right Middle Lobe Collapse",
                       "Right Lower Lobe Collapse", "Left Upper Lobe Collapse",
                       "Left Lower Lobe Collapse"},
            severity_indicators={"partial", "complete"},
            typical_locations={"right upper lobe", "right middle lobe", "right lower lobe",
                               "left upper lobe", "left lower lobe"},
            conflicts_with={"Normal Lung Volume"},
        ))

        # =============================================================
        # 12. PLEURAL DISORDERS
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Pneumothorax",
            synonyms={"PTX", "Air in Pleural Space", "Collapsed Lung"},
            category=FindingCategory.PLEURAL,
            parent_concepts={"Pleural Disorder", "Pleural Air"},
            subtypes={"Simple Pneumothorax", "Spontaneous Pneumothorax",
                       "Traumatic Pneumothorax", "Iatrogenic Pneumothorax"},
            severity_indicators={"small", "moderate", "large"},
            typical_locations={"apical", "lateral", "bilateral"},
            conflicts_with={"Normal Pleural Space"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Tension Pneumothorax",
            synonyms={"Tension PTX", "Tension Pneumo"},
            category=FindingCategory.PLEURAL,
            parent_concepts={"Pleural Disorder", "Pleural Air", "Thoracic Emergency"},
            subtypes=set(),
            severity_indicators={"acute"},
            typical_locations={"hemithorax", "unilateral"},
            conflicts_with={"Normal Pleural Space", "Simple Pneumothorax"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Hemothorax",
            synonyms={"Blood in Pleural Space", "Hemorrhagic Pleural Effusion"},
            category=FindingCategory.PLEURAL,
            parent_concepts={"Pleural Disorder", "Pleural Fluid Collection"},
            subtypes={"Traumatic Hemothorax", "Spontaneous Hemothorax"},
            severity_indicators={"small", "moderate", "large", "massive"},
            typical_locations={"dependent pleural space", "hemithorax"},
            conflicts_with={"Normal Pleural Space"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Pleural Effusion",
            synonyms={"Pleural Fluid", "Fluid in Pleural Space", "Hydrothorax"},
            category=FindingCategory.PLEURAL,
            parent_concepts={"Pleural Disorder", "Pleural Fluid Collection"},
            subtypes={"Transudative Effusion", "Exudative Effusion"},
            severity_indicators={"trivial", "small", "moderate", "large", "massive"},
            typical_locations={"bilateral", "left", "right", "dependent"},
            conflicts_with={"Normal Pleural Space"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Loculated Pleural Effusion",
            synonyms={"Loculated Effusion", "Encysted Effusion", "Trapped Fluid"},
            category=FindingCategory.PLEURAL,
            parent_concepts={"Pleural Disorder", "Pleural Fluid Collection"},
            subtypes=set(),
            severity_indicators={"small", "moderate", "large"},
            typical_locations={"fissural", "lateral", "posterior"},
            conflicts_with={"Free-flowing Pleural Effusion"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Empyema",
            synonyms={"Pleural Empyema", "Infected Pleural Collection", "Pyothorax"},
            category=FindingCategory.PLEURAL,
            parent_concepts={"Pleural Disorder", "Pleural Infection"},
            subtypes=set(),
            severity_indicators={"mild", "moderate", "severe"},
            typical_locations={"pleural space", "hemithorax"},
            conflicts_with={"Simple Pleural Effusion"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Chylothorax",
            synonyms={"Chyle Leak", "Chylous Effusion", "Lymphatic Pleural Effusion"},
            category=FindingCategory.PLEURAL,
            parent_concepts={"Pleural Disorder", "Lymphatic Disorder"},
            subtypes=set(),
            severity_indicators={"small", "moderate", "large"},
            typical_locations={"left hemithorax", "bilateral"},
            conflicts_with={"Normal Pleural Space"},
        ))

        # =============================================================
        # 13. THORACIC MASSES AND NODULES
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Pulmonary Mass",
            synonyms={"Lung Mass", "Pulmonary Lesion"},
            category=FindingCategory.THORACIC_MASSES,
            parent_concepts={"Thoracic Mass", "Pulmonary Neoplasm"},
            subtypes={"Primary Lung Cancer", "Metastatic Mass"},
            severity_indicators={"benign", "suspicious", "malignant"},
            typical_locations={"right upper lobe", "left upper lobe", "central", "peripheral"},
            conflicts_with={"Normal Lung Parenchyma"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Pulmonary Nodule",
            synonyms={"Lung Nodule", "Pulmonary Nodular Lesion", "SPN",
                       "Solitary Pulmonary Nodule"},
            category=FindingCategory.THORACIC_MASSES,
            parent_concepts={"Thoracic Mass", "Pulmonary Finding"},
            subtypes={"Solid Nodule", "Part-Solid Nodule", "Ground Glass Nodule",
                       "Calcified Granuloma"},
            severity_indicators={"benign", "indeterminate", "suspicious"},
            typical_locations={"right upper lobe", "left upper lobe", "bilateral", "peripheral"},
            conflicts_with={"Normal Lung Parenchyma"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Cavitary Lesion",
            synonyms={"Lung Cavity", "Cavitation", "Cavitary Mass", "Cavitary Nodule"},
            category=FindingCategory.THORACIC_MASSES,
            parent_concepts={"Thoracic Mass", "Pulmonary Finding"},
            subtypes={"Thick-Walled Cavity", "Thin-Walled Cavity"},
            severity_indicators={"benign", "suspicious", "malignant"},
            typical_locations={"upper lobes", "bilateral", "peripheral"},
            conflicts_with={"Normal Lung Parenchyma"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Mediastinal Mass",
            synonyms={"Mediastinal Lesion", "Mediastinal Tumor"},
            category=FindingCategory.THORACIC_MASSES,
            parent_concepts={"Thoracic Mass", "Mediastinal Pathology"},
            subtypes={"Anterior Mediastinal Mass", "Middle Mediastinal Mass",
                       "Posterior Mediastinal Mass"},
            severity_indicators={"benign", "indeterminate", "malignant"},
            typical_locations={"anterior mediastinum", "middle mediastinum",
                               "posterior mediastinum", "prevascular"},
            conflicts_with={"Normal Mediastinum"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Mediastinal Lymphadenopathy",
            synonyms={"Enlarged Mediastinal Lymph Nodes", "Mediastinal LAD",
                       "Lymphadenopathy", "Hilar Lymphadenopathy"},
            category=FindingCategory.THORACIC_MASSES,
            parent_concepts={"Mediastinal Pathology", "Lymph Node Disorder"},
            subtypes={"Reactive Lymphadenopathy", "Malignant Lymphadenopathy",
                       "Granulomatous Lymphadenopathy"},
            severity_indicators={"mild", "moderate", "bulky"},
            typical_locations={"paratracheal", "subcarinal", "hilar", "prevascular",
                               "aortopulmonary window"},
            conflicts_with={"Normal-Sized Lymph Nodes"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Thymoma",
            synonyms={"Thymic Tumor", "Thymic Neoplasm"},
            category=FindingCategory.THORACIC_MASSES,
            parent_concepts={"Anterior Mediastinal Mass", "Thymic Pathology"},
            subtypes={"Type A Thymoma", "Type B1 Thymoma", "Type B2 Thymoma", "Type AB Thymoma"},
            severity_indicators={"benign", "invasive", "malignant"},
            typical_locations={"anterior mediastinum", "prevascular space"},
            conflicts_with={"Normal Thymus", "Thymic Hyperplasia"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Thymic Hyperplasia",
            synonyms={"Enlarged Thymus", "Thymic Enlargement", "Rebound Thymic Hyperplasia"},
            category=FindingCategory.THORACIC_MASSES,
            parent_concepts={"Thymic Pathology"},
            subtypes={"True Thymic Hyperplasia", "Lymphoid Thymic Hyperplasia"},
            severity_indicators={"mild", "moderate"},
            typical_locations={"anterior mediastinum"},
            conflicts_with={"Thymoma"},
        ))

        # =============================================================
        # 14. ESOPHAGEAL AND DIAPHRAGMATIC DISORDERS
        # =============================================================
        self._add_entity(ClinicalEntity(
            canonical_name="Esophageal Perforation",
            synonyms={"Esophageal Rupture", "Boerhaave Syndrome", "Esophageal Tear"},
            category=FindingCategory.ESOPHAGEAL_DIAPHRAGMATIC,
            parent_concepts={"Esophageal Pathology", "Gastrointestinal Emergency"},
            subtypes={"Spontaneous Perforation", "Traumatic Perforation", "Iatrogenic Perforation"},
            severity_indicators={"contained", "free", "acute"},
            typical_locations={"distal esophagus", "left posterolateral", "thoracic esophagus"},
            conflicts_with={"Normal Esophagus"},
        ))

        self._add_entity(ClinicalEntity(
            canonical_name="Hiatal Hernia",
            synonyms={"Hiatus Hernia", "Diaphragmatic Hernia", "Sliding Hiatal Hernia",
                       "Paraesophageal Hernia"},
            category=FindingCategory.ESOPHAGEAL_DIAPHRAGMATIC,
            parent_concepts={"Diaphragmatic Disorder", "Esophageal Pathology"},
            subtypes={"Type I Sliding Hernia", "Type II Paraesophageal Hernia",
                       "Type III Mixed Hernia", "Type IV Giant Hernia"},
            severity_indicators={"small", "moderate", "large", "giant"},
            typical_locations={"esophageal hiatus", "posterior mediastinum"},
            conflicts_with={"Normal Diaphragm"},
        ))


# ===================================================================== #
#  CONVENIENCE / MODULE-LEVEL ACCESS
# ===================================================================== #
_default_ontology: Optional[ThoracicOntology] = None


def get_thoracic_ontology() -> ThoracicOntology:
    """Singleton erişimi – ontology'yi ilk çağrıda oluşturur."""
    global _default_ontology
    if _default_ontology is None:
        _default_ontology = ThoracicOntology()
    return _default_ontology


# ===================================================================== #
#  CHEST CT WORKFLOW (modalite-özel pre/post processing)
# ===================================================================== #

class ChestCTWorkflow(BaseModalityWorkflow):
    """
    Chest CT modalitesi için pre/post processing workflow.
    ThoracicOntology'yi kullanarak:
      - Entity normalization (synonym → canonical)
      - Conflict detection (çelişen bulgular)
      - Category-based statistics
    """

    def __init__(self):
        super().__init__(modality="chest_ct")
        self.ontology = get_thoracic_ontology()

    # ------------------------------------------------------------------ #
    #  PRE-PROCESS: evaluation öncesi entity düzeltme / zenginleştirme
    # ------------------------------------------------------------------ #
    def pre_process(self, entities: List[Dict]) -> List[Dict]:
        """
        Entity'leri normalleştir:
        1. general_finding'i ontolojideki canonical name'e çevir
        2. Ontolojiden category bilgisi ekle
        3. Eşleşmeyen (bilinmeyen) entity'leri işaretle
        """
        processed = []
        for ent in entities:
            ent = dict(ent)  # shallow copy — orijinali bozmamak için
            finding = ent.get('observation', '')

            resolved = self.ontology.resolve(finding)
            if resolved:
                ent['_canonical_finding'] = resolved.canonical_name
                ent['_finding_category'] = resolved.category.value
                ent['_ontology_matched'] = True
            else:
                ent['_canonical_finding'] = finding
                ent['_finding_category'] = 'unknown'
                ent['_ontology_matched'] = False

            processed.append(ent)
        return processed

    # ------------------------------------------------------------------ #
    #  POST-PROCESS: evaluation sonrası istatistik ve conflict analizi
    # ------------------------------------------------------------------ #
    def post_process(self, entities: List[Dict], violations: List) -> Tuple[List[Dict], Dict]:
        """
        İşlenmiş entity'ler üzerinde:
        1. Ontoloji eşleşme oranı
        2. Kategori dağılımı
        3. Çelişki (conflict) tespiti
        """
        stats: Dict = {
            'ontology_match_rate': 0.0,
            'category_distribution': {},
            'conflicts_detected': [],
            'unmatched_findings': [],
        }

        if not entities:
            return entities, stats

        # 1. Eşleşme oranı
        matched = sum(1 for e in entities if e.get('_ontology_matched', False))
        stats['ontology_match_rate'] = matched / len(entities)

        # 2. Kategori dağılımı
        cat_counts: Dict[str, int] = {}
        for e in entities:
            cat = e.get('_finding_category', 'unknown')
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        stats['category_distribution'] = cat_counts

        # 3. Bilinmeyen finding'ler
        stats['unmatched_findings'] = [
            e.get('observation', '?')
            for e in entities if not e.get('_ontology_matched', False)
        ]

        # 4. Çelişki tespiti (her entity çifti arasında)
        canonical_names = [e.get('_canonical_finding', '') for e in entities]
        for i in range(len(canonical_names)):
            for j in range(i + 1, len(canonical_names)):
                if self.ontology.are_conflicting(canonical_names[i], canonical_names[j]):
                    stats['conflicts_detected'].append({
                        'entity_a': canonical_names[i],
                        'entity_b': canonical_names[j],
                        'source': 'ontology_conflict'
                    })

        return entities, stats


# ===================================================================== #
#  QUICK SELF-TEST
# ===================================================================== #
if __name__ == "__main__":
    onto = get_thoracic_ontology()

    print("=== THORACIC ONTOLOGY SUMMARY ===")
    for cat, count in onto.summary().items():
        print(f"  {cat}: {count} entities")
    print(f"  TOTAL entities: {len(onto.entity_map)}")
    print(f"  TOTAL synonyms indexed: {len(onto.synonym_index)}")

    # Resolve tests
    test_terms = ["PE", "GGO", "HOCM", "IMH", "AAA", "UIP", "PTX", "DCM", "COP", "MAC"]
    print("\n=== RESOLVE TESTS ===")
    for t in test_terms:
        entity = onto.resolve(t)
        if entity:
            print(f"  '{t}' → {entity.canonical_name} [{entity.category.value}]")
        else:
            print(f"  '{t}' → NOT FOUND")

    # Conflict test
    print("\n=== CONFLICT TESTS ===")
    pairs = [
        ("Dilated Cardiomyopathy", "Hypertrophic Cardiomyopathy"),
        ("UIP", "NSIP"),
        ("Thymoma", "Thymic Hyperplasia"),
        ("Pneumonia", "Pleural Effusion"),  # should be False
    ]
    for a, b in pairs:
        print(f"  {a} ↔ {b}: conflict={onto.are_conflicting(a, b)}")

    # Fuzzy search test
    print("\n=== FUZZY SEARCH: 'aortic' ===")
    for canon, score in onto.fuzzy_search("aortic", threshold=0.4):
        print(f"  {canon}: {score:.2f}")