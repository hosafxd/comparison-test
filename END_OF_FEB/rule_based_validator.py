"""
Forte-Style Rule-Based NLP Processor
Pipeline: Raw Entity → Rule Engine → Validated/Enhanced Entity → Evaluation
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class RuleSeverity(Enum):
    ERROR = "error"           # Kesin hata (Contradiction)
    WARNING = "warning"       # Şüpheli (Review gerektirir)
    INFO = "info"             # Öneri/Otomatik düzeltme

class RuleType(Enum):
    ANATOMICAL = "anatomical"     # Anatomik tutarlılık
    CONSISTENCY = "consistency"   # Mantıksal tutarlılık
    SYNTAX = "syntax"             # Format/grammar
    SEMANTIC = "semantic"         # Anlam tutarlılığı

@dataclass
class RuleViolation:
    rule_id: str
    rule_type: RuleType
    severity: RuleSeverity
    message: str
    entity_index: int
    field: Optional[str] = None
    suggestion: Optional[str] = None

class RuleBasedProcessor:
    """
    Forte-style pipeline processor
    Her rule bir 'processor' gibi çalışır: Input alır, annotation ekler veya düzeltir
    """
    
    def __init__(self, modality: str = "general"):
        self.modality = modality
        self.rules = self._load_rules()
        
    def _load_rules(self):
        """Modaliteye göre rule seti yükle"""
        base_rules = [
            self._rule_anatomical_consistency,      # Rule 1
            self._rule_presence_contradiction,      # Rule 2
            self._rule_degree_severity_logic,       # Rule 3
            self._rule_location_finding_compatibility, # Rule 4
            self._rule_measurement_format           # Rule 5
        ]
        
        # Modalite özel rule'lar (Chest CT için)
        if self.modality == "chest_ct":
            base_rules.extend([
                self._rule_chest_ct_cardiopulmonary_normal,
                self._rule_chest_ct_effusion_location,
                #self._rule_chest_ct_nodule_measurement
            ])
            
        return base_rules
    
    def process(self, entities: List[Dict]) -> Tuple[List[Dict], List[RuleViolation]]:
        """
        Forte Pipeline: Tüm entity'leri işle, violationları topla
        Returns: (İşlenmiş entity'ler, Bulunan violation'lar)
        """
        processed = []
        violations = []
        
        for idx, entity in enumerate(entities):
            # Her rule'i çalıştır
            for rule in self.rules:
                try:
                    entity, viols = rule(entity, idx)
                    violations.extend(viols)
                except Exception as e:
                    print(f"Rule execution error: {e}")
            
            processed.append(entity)
            
        return processed, violations
    
    # =========================================================================
    # RULE 1: Anatomical Consistency (Anatomik Tutarlılık)
    # =========================================================================
    def _rule_anatomical_consistency(self, entity: Dict, idx: int) -> Tuple[Dict, List[RuleViolation]]:
        """
        Chest CT'de 'brain' finding'i olamaz gibi anatomik domain kontrolü
        """
        violations = []
        location = entity.get('location', [])
        finding = entity.get('observation', '')

        # Chest CT için geçersiz anatomik yapılar
        if self.modality == "chest_ct":
            # FIX 3: Chest CT routinely visualizes upper abdomen.
            # Only flag truly out-of-field anatomy (brain, pelvis, extremities).
            # Upper abdominal organs (liver, spleen, kidneys, adrenals) are
            # standard "visualized structures" in chest CT reports.
            invalid_for_chest = {
                'brain', 'cerebral', 'cerebellum', 'intracranial',
                'pelvis', 'pelvic', 'hip', 'femur',
                'knee', 'ankle', 'foot', 'hand', 'wrist', 'elbow',
                'lumbar spine', 'sacrum', 'coccyx',
                'bladder', 'uterus', 'prostate', 'rectum', 'colon',
            }
            loc_set = set(str(l).lower() for l in location if l)

            # Check each location word against invalid set
            flagged_locations = []
            for loc in loc_set:
                if any(inv in loc for inv in invalid_for_chest):
                    flagged_locations.append(loc)

            if flagged_locations:
                violations.append(RuleViolation(
                    rule_id="ANAT-001",
                    rule_type=RuleType.ANATOMICAL,
                    severity=RuleSeverity.ERROR,
                    message=f"Invalid anatomy for Chest CT: {flagged_locations}",
                    entity_index=idx,
                    field="location",
                    suggestion="Check if this is truly a Chest CT scan or wrong modality"
                ))

        return entity, violations
    
    # =========================================================================
    # RULE 2: Presence Contradiction (Varlık Çelişkisi)
    # =========================================================================
    def _rule_presence_contradiction(self, entity: Dict, idx: int) -> Tuple[Dict, List[RuleViolation]]:
        """
        Aynı entity'de hem 'present' hem 'absent' olamaz
        Derecelerde çelişki kontrolü: ['mild', 'severe'] aynı anda olamaz
        """
        violations = []
        presence = str(entity.get('observation_presence', '')).lower()
        degree = entity.get('degree', [])

        # ── PRES-002: Invalid vocabulary ──────────────────────────────────────
        VALID_PRESENCE = {'present', 'absent', 'uncertain'}
        if presence and presence not in VALID_PRESENCE:
            violations.append(RuleViolation(
                rule_id="PRES-002",
                rule_type=RuleType.SYNTAX,
                severity=RuleSeverity.ERROR,
                message=f"Invalid presence value '{entity.get('observation_presence')}'. "
                        f"Must be one of: present, absent, uncertain.",
                entity_index=idx,
                field="observation_presence",
                suggestion=f"Change to one of: {VALID_PRESENCE}"
            ))

        # 2A: Presence çelişkisi (with benign degree filtering)
        if presence in ['present', 'absent']:
            if presence == 'absent' and degree:
                # These degree terms are VALID with "absent" —
                # they describe the normal state or finding type, not severity
                benign_absent_degrees = {
                    'acute', 'chronic', 'subacute',
                    'patent', 'normal', 'stable', 'unchanged',
                    'unremarkable', 'new', 'old', 'remote',
                    'significant', 'clinical', 'suspicious',
                    'upper limits of normal', 'borderline',
                    'trace', 'physiologic',
                }
                
                degree_vals = [str(d).lower() for d in degree if d and d != 'None']
                # Only flag if there are TRUE severity degrees remaining
                true_severity_degrees = [d for d in degree_vals if d not in benign_absent_degrees]
                
                if true_severity_degrees:
                    violations.append(RuleViolation(
                        rule_id="PRES-001",
                        rule_type=RuleType.CONSISTENCY,
                        severity=RuleSeverity.ERROR,
                        message=f"Finding marked 'absent' but has severity degrees: {true_severity_degrees}",
                        entity_index=idx,
                        field="degree",
                        suggestion="Remove severity degrees for absent findings or change presence to 'present'"
                    ))
        
        # 2B: Degree mantığı (mild ve severe aynı anda olamaz)
        if isinstance(degree, list) and len(degree) > 1:
            degree_lower = set(str(d).lower() for d in degree)
            if 'mild' in degree_lower and 'severe' in degree_lower:
                violations.append(RuleViolation(
                    rule_id="DEG-001",
                    rule_type=RuleType.CONSISTENCY,
                    severity=RuleSeverity.ERROR,
                    message="Contradictory degrees: 'mild' and 'severe' together",
                    entity_index=idx,
                    field="degree",
                    suggestion="Select single severity level"
                ))
        
        return entity, violations
    
    # =========================================================================
    # RULE 3: Degree-Severity Logic (Derece-Siddet Mantığı)
    # =========================================================================
    def _rule_degree_severity_logic(self, entity: Dict, idx: int) -> Tuple[Dict, List[RuleViolation]]:
        """
        Finding ile degree uyumu:
        - 'Normal' finding varsa 'severe' degree olamaz
        - 'Trace' sadece certain findings ile kullanılabilir
        """
        violations = []
        finding = str(entity.get('observation', '')).lower()
        degree = [str(d).lower() for d in entity.get('degree', []) if d]
        
        # 'Normal' ve 'severe' çelişkisi
        if 'normal' in finding and any(d in ['severe', 'large', 'extensive'] for d in degree):
            violations.append(RuleViolation(
                rule_id="SEV-001",
                rule_type=RuleType.SEMANTIC,
                severity=RuleSeverity.WARNING,
                message=f"Finding indicates 'normal' but degree suggests severity: {degree}",
                entity_index=idx,
                suggestion="Verify if finding should be 'abnormal' or degree should be 'mild/normal'"
            ))
        
        # 'Trace' kullanımı
        if 'trace' in degree and finding not in ['effusion', 'fluid', 'hemorrhage']:
            violations.append(RuleViolation(
                rule_id="SEV-002",
                rule_type=RuleType.SEMANTIC,
                severity=RuleSeverity.INFO,
                message=f"'Trace' typically used for fluids, applied to: {finding}",
                entity_index=idx,
                suggestion="Consider if 'mild' or 'small' is more appropriate"
            ))
        
        return entity, violations
    
    # =========================================================================
    # RULE 4: Location-Finding Compatibility (Yer-Bulgu Uyumu)
    # =========================================================================
    def _rule_location_finding_compatibility(self, entity: Dict, idx: int) -> Tuple[Dict, List[RuleViolation]]:
        """
        Belirli bulgular sadece belirli yerlerde olabilir:
        - 'Pericardial effusion' → 'pericardium' veya 'heart' location olmalı
        - 'Pleural effusion' → 'pleura' veya 'lung' olmalı
        """
        violations = []
        finding = str(entity.get('observation', '')).lower()
        location = [str(l).lower() for l in entity.get('location', []) if l]
        
        
        # Effusion kontrolü
        if 'effusion' in finding:
            valid_effusion_sites = {'pericardium', 'pericardial', 'pleura', 'pleural', 
                                   'lung', 'lungs', 'abdomen', 'peritoneum'}
            # Substring matching: "left pleural space" should match "pleural"
            if location and not any(
                valid in loc for loc in location for valid in valid_effusion_sites
            ):
                violations.append(RuleViolation(
                    rule_id="LOC-001",
                    rule_type=RuleType.ANATOMICAL,
                    severity=RuleSeverity.ERROR,
                    message=f"Effusion found in invalid location: {location}",
                    entity_index=idx,
                    field="location",
                    suggestion="Effusion must be in body cavity (pericardium, pleura, peritoneum)"
                ))
        
        # Aorta bulguları
        if 'aorta' in finding or 'aortic' in finding:
            aorta_terms = {'aorta', 'aortic', 'ascending', 'descending', 'arch'}
            if location and not any(
                term in loc for loc in location for term in aorta_terms
            ):
                violations.append(RuleViolation(
                    rule_id="LOC-002",
                    rule_type=RuleType.ANATOMICAL,
                    severity=RuleSeverity.WARNING,
                    message=f"Aortic finding without aortic location: {location}",
                    entity_index=idx,
                    suggestion="Add specific aortic location (ascending, descending, arch)"
                ))
        
        return entity, violations
    
    # =========================================================================
    # RULE 5: Measurement Format Validation (Ölçüm Formatı)
    # =========================================================================
    def _rule_measurement_format(self, entity: Dict, idx: int) -> Tuple[Dict, List[RuleViolation]]:

        return entity, []

    # =========================================================================
    # RULE 6 (Chest CT): Cardiopulmonary Normal Contradiction
    # =========================================================================
    def _rule_chest_ct_cardiopulmonary_normal(self, entity: Dict, idx: int) -> Tuple[Dict, List[RuleViolation]]:
        """
        'Normal/unremarkable' finding varsa ama abnormal degree/presence varsa → çelişki.
        """
        violations = []
        finding = str(entity.get('observation', '')).lower()   # ← was 'observation' variable undefined

        normal_keywords = {'normal', 'unremarkable', 'within normal limits', 'no acute'}
        is_normal_finding = any(nk in finding for nk in normal_keywords)  # ← was 'observation', now 'finding'

        degree = [str(d).lower() for d in entity.get('degree', []) if d]
        abnormal_degrees = {'mild', 'moderate', 'severe', 'large', 'extensive', 'massive'}

        if is_normal_finding and any(d in abnormal_degrees for d in degree):
            violations.append(RuleViolation(
                rule_id="CHEST-NORM-001",
                rule_type=RuleType.CONSISTENCY,
                severity=RuleSeverity.ERROR,
                message=f"'Normal/unremarkable' finding has abnormal degree: {degree}",
                entity_index=idx,
                field="degree",
                suggestion="Remove degree or change finding to specific abnormality"
            ))

        return entity, violations

    # =========================================================================
    # RULE 7 (Chest CT): Effusion Type-Location Specificity
    # =========================================================================
    def _rule_chest_ct_effusion_location(self, entity: Dict, idx: int) -> Tuple[Dict, List[RuleViolation]]:
        """
        Placeholder — no violations raised currently.
        Body was empty but referenced undefined 'violations' variable, causing NameError.
        """
        violations = []
        return entity, violations