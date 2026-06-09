"""
Fall Risk Rule-Based Clinical Decision Support System
======================================================
Based on the Evidence Library (All Risk Levels).

Input : Discharge summary (free text) or directly-set PatientVariables
Output: Risk level + evidence-based recommendations

Architecture:
  PatientVariables   -- all variables across primary / secondary / tertiary levels
  VariableExtractor  -- regex NLP to pull structured variables from free text
  RuleEngine         -- applies Evidence Library rules, outputs DecisionOutput
  OutputFormatter    -- human-readable or JSON output

Risk stratification path:
  pass_3kq = True                                         → Low Risk (path A)
  pass_3kq = False, pass_fall_severity = True,
    unsteady_gait = False                                 → Low Risk (path B, same recs)
  pass_3kq = False, pass_fall_severity = True,
    unsteady_gait or poor_gait_balance_muscle_strength    → Intermediate Risk
  pass_3kq = False, pass_fall_severity = False            → High Risk (multifactorial)
"""

import re
import json
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# FRIDs LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

class FRIDsLibrary:
    """
    Loads and indexes fall-risk-increasing drugs (FRIDs) from frids_library.json.
    Returns both a pass/fail flag and category-level matched medication names.

    pass_FRIDs = False  -> one or more FRIDs detected
    pass_FRIDs = None   -> no FRIDs detected in the available text; absence cannot be confirmed
    """

    def __init__(self, json_path: Optional[str] = None):
        self._catalogue = self._load_catalogue(json_path)
        self._patterns = {}
        for category, drugs in self._catalogue.items():
            patterns = []
            for drug in drugs:
                if not drug:
                    continue
                escaped = re.escape(drug.lower()).replace(r"\ ", r"\s+")
                patterns.append((drug, re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)))
            self._patterns[category] = patterns

    def _load_catalogue(self, json_path: Optional[str]) -> dict:
        candidate_paths = []
        if json_path:
            candidate_paths.append(Path(json_path))
        candidate_paths.extend([
            Path(__file__).with_name("frids_library.json"),
            Path.cwd() / "frids_library.json",
        ])

        for path in candidate_paths:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8-sig"))
                    categories = data.get("categories", {})
                    catalogue = {}
                    for key, value in categories.items():
                        label = value.get("label", key)
                        meds = [m.get("name", "").lower() for m in value.get("medications", []) if m.get("name")]
                        if meds:
                            catalogue[label] = sorted(set(meds))
                    if catalogue:
                        return catalogue
                except Exception:
                    pass
        return {}

    def detect(self, text: str) -> tuple[Optional[bool], dict[str, list[str]]]:
        if not text or not self._patterns:
            return None, {}
        matched = {}
        for category, drug_patterns in self._patterns.items():
            for drug_name, pattern in drug_patterns:
                if pattern.search(text):
                    matched.setdefault(category, [])
                    if drug_name not in matched[category]:
                        matched[category].append(drug_name)
        if matched:
            return False, matched
        return None, {}


_FRIDS_LIB = FRIDsLibrary()


def _build_best_frids_library(json_path: Optional[str] = None):
    """
    Prefer the root-level FRIDs library when available because it contains the
    more comprehensive 421-drug catalogue. Fall back to the local JSON-backed
    library if that file cannot be loaded.
    """
    if json_path:
        return FRIDsLibrary(json_path)

    root_module_path = Path(__file__).resolve().parent.parent / "decision_system.py"
    if root_module_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "_root_decision_system_for_frids", root_module_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.FRIDsLibrary()
        except Exception:
            pass

    return _FRIDS_LIB


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientVariables:
    """
    Structured variables mapped to Evidence Library variable scheme.
    None = not yet determined / not extractable from text.
    """

    # ── Primary: Risk Stratification Screening (3 Key Questions) ─────────────
    fall_within_one_year: Optional[int] = None   # ordinal fall-history proxy: 0 no fall history, 1 index fall, 2 recent/repeated falls, None = no fall-related evidence
    feel_unsteady: Optional[bool] = None
    worry_fall: Optional[bool] = None

    # ── Secondary: APSS ───────────────────────────────────────────────────────
    pass_apss: Optional[bool] = None             # True = all APSS answers negative
    have_heart_condition: Optional[bool] = None
    ex_chest_discomfort: Optional[bool] = None
    ex_dizzy: Optional[bool] = None
    ex_asthma: Optional[bool] = None
    has_diabetes: Optional[bool] = None
    exercise_condition: Optional[bool] = None    # other condition requiring exercise caution

    # ── Secondary: Fall Severity Assessment ───────────────────────────────────
    unable_get_up: Optional[bool] = None
    loc_syncope: Optional[bool] = None           # loss of consciousness / suspected syncope
    fall_with_injury: Optional[bool] = None
    is_frail: Optional[bool] = None              # Frailty Phenotype / CFS >= 4

    # ── Secondary: Gait and Balance ───────────────────────────────────────────
    unsteady_gait: Optional[bool] = None         # gait speed <=0.8 m/s or TUG >15 s

    # ── Tertiary: Multifactorial Assessment ───────────────────────────────────
    pass_FRIDs: Optional[bool] = None            # True = NOT on fall-risk-increasing drugs
    detected_FRIDs: dict = field(default_factory=dict)  # {category: [drug_names]}
    concerns_about_falling: Optional[bool] = None  # FES-I moderate+ or expressed fear
    is_delirium: Optional[bool] = None
    is_cognitive_impaired: Optional[bool] = None   # MoCA<26, MMSE<=24, dementia dx, etc.
    have_syncope: Optional[bool] = None
    have_orthostatic_hypo: Optional[bool] = None
    other_cardiac_condition: Optional[bool] = None
    non_terminal_pain: Optional[bool] = None
    terminal_pain: Optional[bool] = None
    with_foot_problem: Optional[bool] = None
    poor_gait_balance_muscle_strength: Optional[bool] = None
    have_vertigo: Optional[bool] = None          # vertigo complaint, no vestibular dx
    have_BPPV: Optional[bool] = None             # positive Dix-Hallpike
    have_vestibular_disease: Optional[bool] = None  # positive head impulse test
    have_eye_problem: Optional[bool] = None
    hearing_impaired: Optional[bool] = None
    urinary_incontinence: Optional[bool] = None
    home_modification_need: Optional[bool] = None
    functional_disabled: Optional[bool] = None   # ADL/IADL <6
    walk_with_aid: Optional[bool] = None
    mal_nutrition: Optional[bool] = None
    vit_D_deficiency: Optional[bool] = None      # serum Vit D <50 nmol/L
    cal_deficiency: Optional[bool] = None        # serum Ca <8.5 mg/dL
    with_fracture_risk: Optional[bool] = None

    # ── Exclusion criteria ────────────────────────────────────────────────────
    fluid_restriction: Optional[bool] = None
    heart_failure: Optional[bool] = None
    have_renal_stone: Optional[bool] = None
    have_hypercalcemia: Optional[bool] = None
    discharge_home: Optional[bool] = None

    # ── Internal ──────────────────────────────────────────────────────────────
    evidence_status: dict = field(default_factory=dict)
    evidence_notes: dict = field(default_factory=dict)
    extraction_notes: list = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def pass_3kq(self) -> Optional[bool]:
        """
        Derive 3 Key Questions result.

        - False: any question is clearly positive
        - True: all three questions are known negative
        - None: no positive answer found, but one or more questions remain unknown
        """
        if self.fall_within_one_year is not None and self.fall_within_one_year > 0:
            return False
        if self.feel_unsteady is True or self.worry_fall is True:
            return False
        if (self.fall_within_one_year == 0 and
                self.feel_unsteady is False and
                self.worry_fall is False):
            return True
        return None

    @property
    def pass_fall_severity(self) -> Optional[bool]:
        """True when fall severity indicators are all negative (non-severe fall)."""
        fall_count_severe = None
        if self.fall_within_one_year is not None:
            fall_count_severe = self.fall_within_one_year >= 2

        indicators = [
            fall_count_severe,
            self.unable_get_up,
            self.loc_syncope,
            self.fall_with_injury,
            self.is_frail,
        ]
        if all(v is None for v in indicators):
            return None
        known = [v for v in indicators if v is not None]
        return not any(known)


@dataclass
class Recommendation:
    category: str
    text: str
    source: str
    grade: str = ""
    is_conditional: bool = False
    condition_note: str = ""


@dataclass
class DecisionOutput:
    risk_level: str
    condition: str
    recommendations: list
    extracted_variables: dict
    warnings: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# VARIABLE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class VariableExtractor:
    """
    Extracts clinical variables from free-text discharge summaries using
    rule-based NLP. All matching is case-insensitive.
    Negative patterns are checked before positive to avoid false positives.
    Tertiary / scored variables that cannot be reliably extracted from free text
    remain None and must be entered directly by a clinician.
    """

    def __init__(self, frids_lib=None):
        self._frids_lib = frids_lib or _FRIDS_LIB

    def extract(self, text: str) -> PatientVariables:
        t = text.lower()
        v = PatientVariables()

        # Primary
        v.fall_within_one_year = self._fall_count(t)
        v.feel_unsteady = self._feel_unsteady(t)
        v.worry_fall = self._worry_fall(t)

        # Secondary: APSS
        v.have_heart_condition = self._heart_condition(t)
        v.ex_chest_discomfort = self._chest_discomfort(t)
        v.ex_dizzy = self._exercise_dizziness(t)
        v.ex_asthma = self._asthma(t)
        v.has_diabetes = self._diabetes(t)
        v.exercise_condition = self._exercise_condition(t)
        v.pass_apss = self._derive_apss(v)

        # Secondary: Fall Severity
        v.unable_get_up = self._unable_get_up(t)
        v.loc_syncope = self._loc_syncope(t)
        v.fall_with_injury = self._fall_with_injury(t)
        v.is_frail = self._frailty(t)

        # Secondary: Gait
        v.unsteady_gait = self._unsteady_gait(t)

        # Exclusion criteria
        v.fluid_restriction = self._fluid_restriction(t)
        v.heart_failure = self._heart_failure(t)
        v.have_renal_stone = self._renal_stone(t)
        v.have_hypercalcemia = self._hypercalcemia(t)  
        v.discharge_home = self._discharge_home(t)

        # Tertiary variables that are reliably mentioned in discharge summaries
        v.is_delirium = self._delirium(t)
        v.is_cognitive_impaired = self._cognitive_impaired(t)
        v.have_syncope = self._syncope(t)
        v.have_orthostatic_hypo = self._orthostatic_hypo(t)
        v.other_cardiac_condition = self._other_cardiac(t)
        v.have_eye_problem = self._eye_problem(t)
        v.hearing_impaired = self._hearing_impaired(t)
        v.urinary_incontinence = self._urinary_incontinence(t)
        v.with_fracture_risk = self._fracture_risk(t)
        v.have_vertigo = self._vertigo(t)
        v.have_BPPV = self._bppv(t)
        v.have_vestibular_disease = self._vestibular_disease(t)
        v.with_foot_problem = self._foot_problem(t)
        v.non_terminal_pain = self._non_terminal_pain(t)
        v.terminal_pain = self._terminal_pain(t)
        v.mal_nutrition = self._malnutrition(t)
        v.concerns_about_falling = self._fear_of_falling(t)
        v.walk_with_aid = self._walk_with_aid(t)
        med_text = self._extract_discharge_med_section(t)
        v.pass_FRIDs, v.detected_FRIDs = self._frids_lib.detect(med_text)
        v.poor_gait_balance_muscle_strength = self._gait_balance(t)
        v.home_modification_need = None
        v.functional_disabled = self._functional_disabled(t)
        v.vit_D_deficiency = self._vit_d_deficiency(t)
        v.cal_deficiency = self._cal_deficiency(t)

        self._record_explicit_evidence(v)
        self._apply_inference_layer(text, t, v, med_text)

        return v

    # ─── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _sentences(text):
        return re.split(r'(?<=[.!?;])\s+|\n+', text)

    @staticmethod
    def _has_sentence_negation(sentence, pos_patterns):
        neg_words = [
            r'\bno\b',
            r'\bdenies?\b',
            r'\bdenied\b',
            r'\bwithout\b',
            r'\bnegative\s+for\b',
            r'\bnot\s+experiencing\b',
            r'\bno\s+evidence\s+of\b',
            r'\bfree\s+of\b'
        ]

        has_neg = any(re.search(n, sentence) for n in neg_words)
        has_pos = any(re.search(p, sentence) for p in pos_patterns)

        return has_neg and has_pos

    @staticmethod
    def _match(patterns, text):
        return any(re.search(p, text) for p in patterns)

    @classmethod
    def _bool_neg_pos(cls, neg_patterns, pos_patterns, text):
        """
        Return False if:
        1. explicit negative pattern matches, OR
        2. negation word and positive keyword appear in the same sentence.

        Return True if positive pattern matches without negation.
        Else None.
        """

        # 1. Explicit variable-specific negation
        if cls._match(neg_patterns, text):
            return False

        # 2. Sentence-level negation
        for sent in cls._sentences(text):
            if cls._has_sentence_negation(sent, pos_patterns):
                return False

        # 3. Positive match
        if cls._match(pos_patterns, text):
            return True

        return None

    @staticmethod
    def _status_for_value(value):
        if value is None:
            return "not_documented_or_not_extractable"
        if isinstance(value, bool):
            return "explicit_positive" if value else "explicit_negative"
        if isinstance(value, int):
            return "explicit_positive" if value > 0 else "explicit_negative"
        return "explicit_value"

    def _record_explicit_evidence(self, v: PatientVariables):
        for key, value in v.__dict__.items():
            if key in ("evidence_status", "evidence_notes", "extraction_notes"):
                continue
            if key == "detected_FRIDs":
                if value:
                    v.evidence_status[key] = "explicit_positive"
                    v.evidence_notes[key] = "FRIDs matched in the discharge medication section."
                else:
                    v.evidence_status[key] = "not_detected"
                continue
            if value is None:
                continue
            v.evidence_status[key] = self._status_for_value(value)

    def _set_inferred(self, v: PatientVariables, attr: str, value, note: str):
        if getattr(v, attr) is not None:
            return
        setattr(v, attr, value)
        if isinstance(value, bool):
            status = "inferred_positive" if value else "inferred_negative"
        elif isinstance(value, int):
            status = "inferred_positive" if value > 0 else "inferred_negative"
        else:
            status = "inferred_value"
        v.evidence_status[attr] = status
        v.evidence_notes[attr] = note

    def _apply_inference_layer(self, original_text: str, t: str, v: PatientVariables, med_text: str):
        sections = self._build_section_context(t)

        self._infer_pass_frids(v, med_text)
        self._infer_walk_with_aid(v, sections)
        self._infer_unsteady_gait(v, sections)
        self._infer_functional_disabled(v, sections)
        self._infer_home_modification_need(v, sections)
        self._infer_other_cardiac_condition(v, sections)
        self._infer_fall_screen_questions(v, sections)

        self._finalise_evidence_status(v)

    def _finalise_evidence_status(self, v: PatientVariables):
        for key, value in v.__dict__.items():
            if key in ("evidence_status", "evidence_notes", "extraction_notes"):
                continue
            if key == "detected_FRIDs":
                if key not in v.evidence_status:
                    v.evidence_status[key] = "not_detected"
                continue
            v.evidence_status.setdefault(key, "not_documented_or_not_extractable")

    def _build_section_context(self, t: str) -> dict:
        def _slice(start_patterns, end_patterns):
            start = None
            for pattern in start_patterns:
                match = re.search(pattern, t)
                if match:
                    start = match.end()
                    break
            if start is None:
                return ""
            remaining = t[start:]
            end_positions = []
            for pattern in end_patterns:
                match = re.search(pattern, remaining)
                if match:
                    end_positions.append(match.start())
            if end_positions:
                return remaining[:min(end_positions)]
            return remaining

        section_endings = [
            r'\n[a-z][a-z /&]{2,}:\s*',
            r'chief complaint:',
            r'history of present illness:',
            r'past medical history:',
            r'physical exam:',
            r'discharge medications?:',
            r'discharge disposition:',
            r'followup instructions:',
            r'hospital course:',
            r'assessment and plan:',
        ]
        return {
            "disposition": _slice([r'discharge disposition:'], section_endings),
            "hospital_course": _slice([r'hospital course:'], section_endings),
            "hpi": _slice([r'history of present illness:'], section_endings),
            "physical_exam": _slice([r'physical exam:'], section_endings),
            "pmh": _slice([r'past medical history:'], section_endings),
            "therapy": _slice([r'physical therapy:', r'pt:'], section_endings),
            "medications": self._extract_discharge_med_section(t),
            "full_text": t,
        }

    def _infer_pass_frids(self, v: PatientVariables, med_text: str):
        if v.pass_FRIDs is None and med_text.strip():
            self._set_inferred(
                v,
                "pass_FRIDs",
                True,
                "No FRIDs were matched within a non-empty discharge medication section.",
            )

    def _infer_walk_with_aid(self, v: PatientVariables, sections: dict):
        combined = "\n".join([
            sections.get("therapy", ""),
            sections.get("physical_exam", ""),
            sections.get("hospital_course", ""),
            sections.get("disposition", ""),
            sections.get("full_text", ""),
        ])
        if v.walk_with_aid is None:
            if self._match([
                r'\bassist\s*x\s*[12]\b',
                r'\bone\s+person\s+assist\b',
                r'\btwo\s+person\s+assist\b',
                r'\bwith\s+(?:a\s+)?walker\b',
                r'\bwith\s+(?:a\s+)?cane\b',
                r'\buses?\s+(?:a\s+)?walker\b',
                r'\buses?\s+(?:a\s+)?cane\b',
                r'\brequires?\s+(?:a\s+)?walker\b',
                r'\brequires?\s+(?:a\s+)?cane\b',
                r'\bwheelchair\b',
            ], combined):
                self._set_inferred(v, "walk_with_aid", True, "Mobility assistance language suggests a walking aid is required.")
            elif self._match([
                r'\bwalks?\s+independently\b',
                r'\bindependent\s+ambulation\b',
                r'\bwithout\s+aid\b',
                r'\bunaided\b',
            ], combined):
                self._set_inferred(v, "walk_with_aid", False, "Mobility language suggests independent ambulation without a walking aid.")

    def _infer_unsteady_gait(self, v: PatientVariables, sections: dict):
        combined = "\n".join([
            sections.get("therapy", ""),
            sections.get("physical_exam", ""),
            sections.get("hospital_course", ""),
            sections.get("full_text", ""),
        ])
        if v.unsteady_gait is None:
            if (
                v.walk_with_aid is True
                or v.poor_gait_balance_muscle_strength is True
                or self._match([r'\bassist\s*x\s*[12]\b', r'\bneeds?\s+assistance\s+walking\b'], combined)
            ):
                self._set_inferred(v, "unsteady_gait", True, "Mobility assistance or balance impairment suggests unsteady gait.")
            elif (
                v.walk_with_aid is False
                and self._match([r'\bwalks?\s+independently\b', r'\bindependent\s+ambulation\b'], combined)
            ):
                self._set_inferred(v, "unsteady_gait", False, "Independent ambulation language suggests gait is not unsteady.")

    def _infer_functional_disabled(self, v: PatientVariables, sections: dict):
        combined = "\n".join([
            sections.get("therapy", ""),
            sections.get("disposition", ""),
            sections.get("hospital_course", ""),
            sections.get("full_text", ""),
        ])
        if v.functional_disabled is None:
            if (
                v.walk_with_aid is True
                or v.unsteady_gait is True
                or self._match([
                    r'\bassist\s*x\s*[12]\b',
                    r'\bneeds?\s+assistance\s+with\s+(?:mobility|transfers|ambulation|adl)\b',
                    r'\brehab\b',
                    r'\bskilled\s+nursing\b',
                    r'\bsnf\b',
                    r'\bhome\s+with\s+services\b',
                ], combined)
            ):
                self._set_inferred(v, "functional_disabled", True, "Disposition or mobility context suggests functional limitation.")
            elif self._match([
                r'\bwalks?\s+independently\b',
                r'\bindependent\s+with\s+adl\b',
                r'\bactivity\s+status\s*:\s*ambulatory\s*-\s*independent\b',
            ], combined):
                self._set_inferred(v, "functional_disabled", False, "Documentation suggests independence with ambulation or ADLs.")

    def _infer_home_modification_need(self, v: PatientVariables, sections: dict):
        if v.home_modification_need is None:
            if any([
                v.have_eye_problem is True,
                v.functional_disabled is True,
                v.unsteady_gait is True,
                v.walk_with_aid is True,
            ]):
                self._set_inferred(
                    v,
                    "home_modification_need",
                    True,
                    "Mobility or sensory risk factors suggest home/environment modification should be considered.",
                )

    def _infer_other_cardiac_condition(self, v: PatientVariables, sections: dict):
        combined = "\n".join([sections.get("pmh", ""), sections.get("hpi", ""), sections.get("full_text", "")])
        if v.other_cardiac_condition is None:
            if v.heart_failure is True:
                self._set_inferred(v, "other_cardiac_condition", True, "Heart failure implies a relevant cardiac comorbidity.")
            elif v.have_syncope is True and self._match([
                r'\bafib\b',
                r'\batrial\s+fibrillation\b',
                r'\bpacemaker\b',
                r'\bheart\s+block\b',
                r'\barrhythmia\b',
            ], combined):
                self._set_inferred(v, "other_cardiac_condition", True, "Syncope plus cardiac history suggests another cardiac condition.")

    def _infer_fall_screen_questions(self, v: PatientVariables, sections: dict):
        combined = "\n".join([
            sections.get("hpi", ""),
            sections.get("hospital_course", ""),
            sections.get("full_text", ""),
        ])
        if v.fall_within_one_year is None:
            fall_count = self._fall_count(combined)
            if fall_count is not None:
                self._set_inferred(
                    v,
                    "fall_within_one_year",
                    fall_count,
                    "Presentation context supports ordinal fall-history classification.",
                )

        if v.feel_unsteady is None and v.unsteady_gait is True:
            self._set_inferred(v, "feel_unsteady", True, "Observed gait instability suggests the patient likely feels unsteady.")

        if v.feel_unsteady is None and v.unsteady_gait is False and v.functional_disabled is False:
            if self._match([r'\bwalks?\s+independently\b', r'\bindependent\s+ambulation\b'], combined):
                self._set_inferred(v, "feel_unsteady", False, "Independent ambulation without gait concerns suggests no documented unsteadiness.")

    @classmethod
    def _same_sentence_match(cls, text, group1_patterns, group2_patterns):
        for sent in cls._sentences(text):
            if cls._has_sentence_negation(sent, group1_patterns):
                continue
            if cls._has_sentence_negation(sent, group2_patterns):
                continue

            if cls._match(group1_patterns, sent) and cls._match(group2_patterns, sent):
                return True

        return False
    
    def _extract_discharge_med_section(self, t: str) -> str:
        """
        Extract only the discharge medication section from discharge summary.
        Returns empty string if section is not found.
        """

        section_patterns = [
            r'\bdischarge\s+medications?\s*:'
        ]


        end_patterns = [
            r'\bdischarge\s+disposition\s*:',
            r'\bdischarge\s+diagnosis\s*:',
            r'\bdischarge\s+instructions\s*:',
            r'\bfollow\s*up\s+instructions\s*:',
            r'\bfollowup\s+instructions\s*:'
        ]


        start_match = None
        for pattern in section_patterns:
            m = re.search(pattern, t)
            if m:
                start_match = m
                break

        if not start_match:
            return ""

        section_start = start_match.end()
        remaining_text = t[section_start:]

        end_positions = []
        for pattern in end_patterns:
            m = re.search(pattern, remaining_text)
            if m:
                end_positions.append(m.start())

        if end_positions:
            section_end = min(end_positions)
            return remaining_text[:section_end]

        return remaining_text

    # ─── Fall count ───────────────────────────────────────────────────────────
    def _fall_count(self, t):
        no_fall = [
            r'\bno\s+(?:known\s+)?(?:history\s+of\s+)?falls?\b',
            r'\bden(?:ies|y|ied)\s+(?:any\s+)?(?:history\s+of\s+)?falls?\b',
            r'\bno\s+falls?\s+(?:reported|noted|documented|in\s+the\s+past)\b',
            r'\bfalls?\s*:\s*(?:no|none|nil|negative)\b',
            r'\bhas\s+not\s+(?:fallen|had\s+a\s+fall)\b',
            r'\bfall[s]?\s+(?:screen(?:ing)?\s*)?(?:negative|nil)\b',
            r'\bno\s+falls?\s+(?:in|within|over)\s+(?:the\s+)?(?:past|last|previous)\s+(?:year|12\s*months|one\s+year)\b',
        ]
        if self._match(no_fall, t):
            return 0

        recent_or_repeated_fall = [
            r'\b(?:recent|recently|prior|previous|subsequent)\s+falls?\b',
            r'\bfalls?\s+(?:recently|again|repeatedly)\b',
            r'\b(?:recurrent|repeated|multiple|frequent|several)\s+falls?\b',
            r'\b(?:two|three|four|five|six|seven|eight|nine|ten|[2-9]\d*)\s+falls?\b',
            r'\bfalls?\s+(?:in|within|over)\s+(?:the\s+)?(?:last|past|previous)\s+(?:year|12\s*months|one\s+year)\b',
            r'\bhistory\s+of\s+(?:recurrent|repeated|multiple|frequent)?\s*falls?\b',
            r'\bfell\s+(?:recently|again|last\s+year|this\s+year|previously)\b',
        ]
        if self._match(recent_or_repeated_fall, t):
            return 2

        admit_for_fall = [
            r'\b(?:admitted|admission|presented|presentation|seen|evaluated|brought\s+in)(?:\s+\w+){0,6}\s+(?:for|after|following|due\s+to|because\s+of|with)\s+(?:an?\s+)?fall\b',
            r'\b(?:mechanical|ground[-\s]level|unwitnessed|witnessed)\s+fall\b',
            r'\bglf\b',
            r'\bslipped\s+and\s+fell\b',
            r'\btrip(?:ped)?\s+and\s+fell\b',
            r'\bfall(?:ed|s)?\s+(?:at\s+home|at\s+rest\s+home|from\s+standing|from\s+bed|from\s+chair)\b',
            r'\bfound\s+on\s+(?:the\s+)?floor\b',
        ]
        if self._match(admit_for_fall, t):
            return 1

        return None

    # ─── Feels unsteady ───────────────────────────────────────────────────────
    def _feel_unsteady(self, t):
        return self._bool_neg_pos(
            [r'no\s+(?:balance|gait)\s+(?:problems?|issues?)',
             r'steady\s+(?:gait|on\s+(?:his|her|their)?\s*feet)',
             r'(?:good|normal|intact|stable)\s+balance',
             r'den(?:ies|y)\s+(?:feeling\s+)?unsteady',
             r'does\s+not\s+feel\s+unsteady',
             r'not\s+(?:feeling|feel)\s+unsteady',
             r'gait\s*:\s*(?:normal|stable|intact)'],
            [r'(?:feels?|feeling|reports?|complains?\s+of)\s+unsteady',
             r'unsteady\s+(?:on\s+(?:his|her|their)?\s*feet|gait|balance)',
             r'balance\s+(?:problems?|issues?|difficulties?|impairment)',
             r'gait\s+(?:instability|unsteady|abnormal|ataxia)',
             r'(?:poor|impaired|reduced)\s+balance',
             r'difficulty\s+(?:walking|ambulating|mobilising)',
             r'ataxic\s+gait\b'], t)

    # ─── Worry about falling ──────────────────────────────────────────────────
    def _worry_fall(self, t):
        return self._bool_neg_pos(
            [r'(?:no|not)\s+(?:worry|fear|concern)\s+(?:about\s+)?falls?',
             r'not\s+(?:worried|concerned|afraid)\s+(?:about\s+)?falls?',
             r'den(?:ies|y)\s+(?:any\s+)?(?:fear|worry|anxiety)\s+(?:of|about\s+)?fall(?:ing)?\b',
             r'no\s+fear\s+of\s+fall(?:ing)?\b'],
            [r'(?:worry|worries|worried|fear|afraid|anxious|concern(?:ed)?)\s+(?:about\s+)?fall(?:ing)?\b',
             r'fear\s+of\s+fall(?:ing)?\b',
             r'fall[- ]related\s+(?:anxiety|fear|concern)'], t)

    # ─── APSS component variables ─────────────────────────────────────────────
    def _heart_condition(self, t):
        heart_vars = [
            self._syncope(t),
            self._orthostatic_hypo(t),
            self._other_cardiac(t)
        ]

        if any(x is True for x in heart_vars):
            return True
        if all(x is False for x in heart_vars):
            return False
        return None

    def _chest_discomfort(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+chest\s+(?:pain|discomfort|tightness|pressure|heaviness|ache)\b',
                r'\bden(?:ies|y|ied)\s+chest\s+(?:pain|discomfort|tightness|pressure|heaviness|ache)\b',
                r'\bwithout\s+chest\s+(?:pain|discomfort|tightness|pressure|heaviness|ache)\b',
                r'\bnegative\s+for\s+chest\s+(?:pain|discomfort|tightness|pressure|heaviness|ache)\b',
                r'\bno\s+(?:angina|anginal\s+pain|anginal\s+symptoms)\b',
                r'\bden(?:ies|y|ied)\s+(?:angina|anginal\s+pain|anginal\s+symptoms)\b'],
            pos_patterns=[
                r'\bchest\s+pain\b',
                r'\bchest\s+discomfort\b',
                r'\bchest\s+tightness\b',
                r'\bchest\s+pressure\b',
                r'\bchest\s+heaviness\b',
                r'\bchest\s+ache\b',
                r'\bangina\b',
                r'\banginal\s+pain\b',
                r'\banginal\s+symptoms\b',
                r'\bexertional\s+chest\s+pain\b',
                r'\bexertional\s+chest\s+discomfort\b',
                r'\bsubsternal\s+chest\s+pain\b',
                r'\bretrosternal\s+chest\s+pain\b',
                r'\batypical\s+chest\s+pain\b',
                r'\bcardiac\s+chest\s+pain\b',
                r'\bischemic\s+chest\s+pain\b'
            ],
            text=t
        )

    def _exercise_dizziness(self, t):
        return self._bool_neg_pos(
            neg_patterns = [
                r'no\s+(?:dizziness|syncope|faint(?:ing)?)\s+(?:on|during|with)\s+(?:exercise|exertion|activity)',
                r'den(?:ies|y)\s+(?:exercise[-\s]induced)?\s+(?:dizziness|syncope)'],
            pos_patterns = [
                r'(?:dizziness|syncope|faint(?:ing)?|los(?:e|t|ing)\s+balance)\s+(?:on|during|with)\s+(?:exercise|exertion|activity)',
                r'exercise[-\s]induced\s+(?:dizziness|syncope|faintness)'
            ],
            text=t
        )

    def _asthma(self, t):
        return self._bool_neg_pos(
            neg_patterns = [
                r'no\s+(?:history\s+of\s+)?asthma\b',
                r'asthma\s*:\s*(?:nil|no|none|resolved)'
            ],
            pos_patterns = [
                r'asthma\s+attack\s+(?:requiring|needing)',
                r'acute\s+asthma\b',
                r'asthma\s+exacerbation'
            ],
            text=t
        )

    def _diabetes(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+(?:history\s+of\s+)?diabetes\b',
                r'\bden(?:ies|y|ied)\s+(?:diabetes|dm|t1dm|t2dm)\b',
                r'\bwithout\s+(?:diabetes|dm|t1dm|t2dm)\b',
                r'\bnegative\s+for\s+(?:diabetes|dm|t1dm|t2dm)\b',
                r'\bno\s+(?:hyperglycemia|hyperglycaemia)\b',
                r'\bno\s+elevated\s+glucose\b',
                r'\bno\s+high\s+blood\s+sugar\b'],
            pos_patterns=[
                r'\bdiabetes\b',
                r'\bdiabetic\b',
                r'\bdm\b',
                r'\bdm1\b',
                r'\bdm2\b',
                r'\bt1dm\b',
                r'\bt2dm\b',
                r'\btype\s+1\s+diabetes\b',
                r'\btype\s+2\s+diabetes\b',
                r'\bpoor\s+glycemic\s+control\b',
                r'\bpoor\s+glycaemic\s+control\b',
                r'\bpoor\s+glucose\s+control\b',
                r'\bpoor\s+blood\s+sugar\s+control\b',
                r'\buncontrolled\s+diabetes\b',
                r'\bhyperglycemia\b',
                r'\bhyperglycaemia\b',
                r'\belevated\s+glucose\b',
                r'\bhigh\s+blood\s+sugar\b',
                r'\bdifficulty\s+controlling\s+blood\s+sugar\b',
                r'\bdifficulty\s+controlling\s+glucose\b',
                r'\bfrequent\s+hyperglycemia\b',
                r'\bfrequent\s+hyperglycaemia\b',
                r'\bfsbg\s+elevated\b',
                r'\bbg\s+elevated\b',
                r'\bglucose\s+poorly\s+controlled\b',
                r'\binsulin[-\s]?dependent\s+diabetes\b',
                r'\bon\s+insulin\s+sliding\s+scale\b',
                r'\bpoorly\s+controlled\s+dm\b',
                r'\bdka\b'
            ],
            text=t
        )
    
    def _exercise_condition(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+exercise\s+intolerance\b',
                r'\bden(?:ies|y|ied)\s+exercise\s+intolerance\b',
                r'\bno\s+(?:activity|exercise)\s+restriction\b',
                r'\bno\s+mobility\s+limitation\b',
                r'\bno\s+functional\s+limitation\b',
                r'\bno\s+physical\s+limitation\b',
                r'\bnot\s+frail\b',
                r'\bno\s+frailty\b',
                r'\bnot\s+deconditioned\b',
                r'\bno\s+deconditioning\b',
                r'\bsafe\s+to\s+ambulate\s+independently\b',
                r'\bindependent\s+ambulation\b',
                r'\bambulat(?:e|es|ing|ed)\s+independently\b',
                r'\bno\s+special\s+precautions?\s+for\s+exercise\b',
                r'\bno\s+exercise\s+precautions?\b',
                r'\bno\s+(?:history\s+of\s+)?parkinson(?:\'s)?\s+disease\b',
                r'\bden(?:ies|y|ied)\s+parkinson(?:\'s)?\s+disease\b',
                r'\bno\s+parkinsonism\b'
            ],
            pos_patterns=[
                r'\bexercise\s+intolerance\b',
                r'\blimited\s+exercise\s+tolerance\b',
                r'\bpoor\s+exercise\s+tolerance\b',
                r'\breduced\s+exercise\s+tolerance\b',
                r'\bactivity\s+intolerance\b',
                r'\bexertional\s+symptoms\b',
                r'\bdeconditioned\b',
                r'\bseverely\s+decondition\b',
                r'\bfrailty\b',
                r'\bfrail\b',
                r'\bfall\s+risk\s+with\s+exercise\b',
                r'\bunsafe\s+to\s+ambulate\s+independently\b',
                r'\brequires?\s+supervised\s+exercise\b',
                r'\bneeds?\s+supervision\s+with\s+mobility\b',
                r'\bexercise\s+precautions?\b',
                r'\bspecial\s+precautions?\s+for\s+exercise\b',
                r'\bmedical\s+clearance\s+for\s+exercise\b',
                r'\bnot\s+safe\s+for\s+unsupervised\s+exercise\b',
                r'\bmobility\s+limitation\b',
                r'\bweight\s+bearing\s+as\s+tolerated\b',
                r'\bwbat\b',
                r'\bnon[-\s]?weight\s+bearing\b',
                r'\bnon\s+weight\s+bearing\b',
                r'\bnwb\b',
                r'\bpartial\s+weight\s+bearing\b',
                r'\bpwb\b',
                r'\bexercise\s+restricted\b',
                r'\bactivity\s+restricted\b',
                r'\bphysical\s+limitation\b',
                r'\bfunctional\s+limitation\b',
                r'\bparkinson(?:\'s)?\s+disease\b',
                r'\bparkinsonism\b',
                r'\bfracture\b',
                r'post[-\s]?op(?:erative)?\s+exercise\s+restriction\b',
            ],
            text=t
        )


    def _derive_apss(self, v: PatientVariables) -> Optional[bool]:
        """
        APSS passes = False if any APSS component is True.
        APSS passes = True if all components are False or None.
        """

        components = [
            v.have_heart_condition,
            v.ex_chest_discomfort,
            v.ex_dizzy,
            v.ex_asthma,
            v.has_diabetes,
            v.exercise_condition
        ]

        if any(c is True for c in components):
            return False

        return True

    # ─── Fall severity variables ──────────────────────────────────────────────
    def _unable_get_up(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bable\s+to\s+(?:get\s+up|rise|stand)\b',
                r'\bable\s+to\s+get\s+off\s+(?:the\s+)?floor\b',
                r'\bable\s+to\s+stand\s+(?:back\s+)?up\b',
                r'\bno\s+difficulty\s+(?:getting\s+up|rising|standing)\b',
                r'\bgot\s+up\s+independently\b',
                r'\brose\s+independently\b',
                r'\bstood\s+up\s+independently\b'
            ],
            pos_patterns=[
                r'\bunable\s+to\s+get\s+up\b',
                r'\bcould\s+not\s+get\s+up\b',
                r'\bcouldn\'?t\s+get\s+up\b',
                r'\bunable\s+to\s+rise\b',
                r'\bunable\s+to\s+stand\s+after\s+fall\b',
                r'\bunable\s+to\s+get\s+off\s+(?:the\s+)?floor\b',
                r'\bcould\s+not\s+get\s+off\s+(?:the\s+)?floor\b',
                r'\bcouldn\'?t\s+get\s+off\s+(?:the\s+)?floor\b',
                r'\bfound\s+on\s+(?:the\s+)?floor\b',
                r'\blay(?:ing)?\s+on\s+(?:the\s+)?floor\b',
                r'\blying\s+on\s+(?:the\s+)?floor\b',
                r'\bdown\s+on\s+(?:the\s+)?floor\b',
                r'\blong\s+lie\s+on\s+(?:the\s+)?floor\b',
                r'\blong\s+lie\s+after\s+fall\b',
                r'\bunable\s+to\s+rise\s+independently\b',
                r'\bunable\s+to\s+mobilize\s+after\s+fall\b',
                r'\bunable\s+to\s+mobilise\s+after\s+fall\b',
                r'\brequired\s+assistance\s+to\s+get\s+up\b',
                r'\bneeded\s+help\s+getting\s+up\b',
                r'\bneeded\s+assistance\s+off\s+(?:the\s+)?floor\b',
                r'\bunable\s+to\s+stand\s+back\s+up\b'
            ],
            text=t
        )

    def _loc_syncope(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+(?:loss\s+of\s+consciousness|loc)\b',
                r'\bden(?:ies|y|ied)\s+(?:loss\s+of\s+consciousness|loc|syncope|syncopal\s+episode)\b',
                r'\bwithout\s+(?:loss\s+of\s+consciousness|loc|syncope)\b',
                r'\bnegative\s+for\s+(?:loss\s+of\s+consciousness|loc|syncope)\b',
                r'\bremained\s+conscious\b',
                r'\bno\s+(?:fainting|faint|blackout|black\s+out)\b',
                r'\bden(?:ies|y|ied)\s+(?:fainting|faint|blackout|black\s+out)\b',
                r'\bno\s+unexplained\s+fall\b'
            ],
            pos_patterns=[
                r'\bsyncope\b',
                r'\bsyncopal\b',
                r'\bpresyncope\b',
                r'\bpre[-\s]?syncope\b',
                r'\bfaint\b',
                r'\bfainted\b',
                r'\bfainting\b',
                r'\bblackout\b',
                r'\bblack(?:ed)?\s+out\b',
                r'\bloss\s+of\s+consciousness\b',
                r'\bloc\b',
                r'\bunwitnessed\s+fall\b',
                r'\bunexplained\s+fall\b'
            ],
            text=t
        )

    def _fall_with_injury(self, t):
        neg_patterns = [
            r'\bno\s+(?:significant\s+)?injur(?:y|ies)\b',
            r'\buninjured\b',
            r'\bfall\s+without\s+injur(?:y|ies)\b',
            r'\bno\s+injur(?:y|ies)\s+(?:after|following|from)\s+(?:the\s+)?fall\b',
            r'\bden(?:ies|y|ied)\s+(?:any\s+)?injur(?:y|ies)\s+(?:after|following|from)\s+(?:the\s+)?fall\b'
        ]

        direct_pos_patterns = [
            r'\bfall\s+with\s+injur(?:y|ies)\b',
            r'\binjur(?:y|ies)\s+after\s+fall\b',
            r'\binjured\s+in\s+(?:the\s+)?fall\b',
            r'\bs/p\s+fall\s+with\s+injur(?:y|ies)\b',
            r'\bpost[-\s]?fall\s+injur(?:y|ies)\b',
            r'\bfall\s+resulting\s+in\s+injur(?:y|ies)\b',
            r'\bfall\s+causing\s+injur(?:y|ies)\b',
            r'\brequired\s+(?:medical\s+)?(?:attention|review|treatment)\s+(?:after|following|for)\s+(?:the\s+)?fall\b',
            r'\badmitted\s+after\s+fall\b',
            r'\bhospitali[sz]ed\s+after\s+fall\b'
        ]

        fall_patterns = [
            r'\bfalls?\b',
            r'\bfell\b',
            r'\bfalling\b',
            r'\bmechanical\s+fall\b',
            r'\bground\s+level\s+fall\b',
            r'\bglf\b',
            r'\brecurrent\s+falls?\b',
            r'\brepeat\s+falls?\b',
            r'\bfrequent\s+falls?\b',
            r'\bhistory\s+of\s+falls?\b',
            r'\bhx\s+of\s+falls?\b',
            r'\bs/p\s+fall\b',
            r'\bpost[-\s]?fall\b',
            r'\bfall\s+at\s+home\b',
            r'\bfall\s+at\s+nursing\s+home\b',
            r'\btrip(?:ped)?\s+and\s+(?:fall|fell)\b',
            r'\bslipped\s+and\s+fell\b',
            r'\bunwitnessed\s+fall\b',
            r'\bwitnessed\s+fall\b',
            r'\bfound\s+on\s+(?:the\s+)?floor\b',
            r'\brecent\s+fall\b'
        ]

        injury_patterns = [
            r'\binjur(?:y|ies|ed)\b',
            r'\btrauma\b',
            r'\bfracture\b',
            r'\bfx\b',
            r'\bbroken\b',
            r'\bbruise\b',
            r'\bbruising\b',
            r'\bhematoma\b',
            r'\bhaematoma\b',
            r'\blaceration\b',
            r'\bcut\b',
            r'\bhead\s+strike\b',
            r'\bhead\s+injur(?:y|ies)\b',
            r'\bintracranial\s+h(?:a)?emorrhage\b',
            r'\bich\b',
            r'\bsubdural\s+h(?:a)?ematoma\b',
            r'\bsdh\b',
            r'\bpain\s+after\s+fall\b',
            r'\bhip\s+fracture\b',
            r'\bfemoral\s+neck\s+fracture\b',
            r'\bwrist\s+fracture\b',
            r'\brib\s+fracture\b',
            r'\bconsulted\s+physician\b',
            r'\bed\s+visit\b',
            r'\bemergency\s+department\b',
            r'\bhospitali[sz]ed\b',
            r'\badmitted\s+after\s+fall\b'
        ]

        if self._match(neg_patterns, t):
            return False

        if self._match(direct_pos_patterns, t):
            return True

        if self._same_sentence_match(t, fall_patterns, injury_patterns):
            return True

        return None

    def _frailty(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bnot\s+frail\b',
                r'\bno\s+frailty\b',
                r'\bden(?:ies|y|ied)\s+frailty\b',
                r'\b(?:robust|pre[-\s]?frail|fit)\b',
                r'\bcfs\s*[:\-]?\s*[1-3]\b',
                r'\bclinical\s+frailty\s+scale\s*[:\-]?\s*[1-3]\b',
                r'\bnot\s+(?:bed\s*bound|bedbound|wheelchair\s*bound|immobile|deconditioned)\b',
                r'\bno\s+(?:failure\s+to\s+thrive|ftt|cachexia|deconditioning)\b'
            ],
            pos_patterns=[
                r'\bfrail\b',
                r'\bfrailty\b',
                r'\bfrail\s+elderly\b',
                r'\bappears\s+frail\b',
                r'\bclinically\s+frail\b',
                r'\bcfs\s*[:\-]?\s*(?:[4-9]|[1-9][0-9]+)\b',
                r'\bclinical\s+frailty\s+scale\s*[:\-]?\s*(?:[4-9]|[1-9][0-9]+)\b',
                r'\bcfs\s*(?:>=|≥|greater\s+than\s+or\s+equal\s+to)\s*4\b',
                r'\bfrailty\s+(?:score|scale|index)\s*[:\-]?\s*(?:[4-9]|[1-9][0-9]+)\b',
                r'\bbed\s*bound\b',
                r'\bbedbound\b',
                r'\bwheelchair\s*bound\b',
                r'\bimmobile\b',
                r'\bfailure\s+to\s+thrive\b',
                r'\bftt\b',
                r'\bcachexia\b',
                r'\bdeconditioned\b',
                r'\bdeconditioning\b'
            ],
            text=t
        )

    # ─── Gait ─────────────────────────────────────────────────────────────────
    def _unsteady_gait(self, t):
        return self._bool_neg_pos(
            [r'normal\s+gait\b',
             r'steady\s+gait\b',
             r'tug\s*[:\-]?\s*(?:[1-9]|1[0-4])\s*s',
             r'gait\s+speed\s*[:\-]?\s*(?:0\.[9-9]|[1-9])'],
            [r'unsteady\s+gait\b',
             r'gait\s+(?:instability|disturbance|impairment|abnormality)',
             r'tug\s*[:\-]?\s*(?:1[5-9]|[2-9]\d)\s*s',
             r'gait\s+speed\s*[:\-]?\s*0\.[0-8]',
             r'ataxic\s+gait\b'], t)

    # ─── Exclusion criteria ───────────────────────────────────────────────────
    def _fluid_restriction(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+fluid\s+restriction\b',
                r'\bnot\s+on\s+fluid\s+restriction\b',
                r'\bwithout\s+fluid\s+restriction\b',
                r'\bfluid\s+restriction\s+(?:not\s+required|not\s+needed|ceased|stopped|discontinued)\b',
                r'\bno\s+(?:esrd|dialysis)\b'
            ],
            pos_patterns=[
                r'\bfluid\s+restriction\b',
                r'\bfluid\s+restricted\b',
                r'\brestrict\s+fluids\b',
                r'\brestricted\s+fluids\b',
                r'\bfree\s+water\s+restriction\b',
                r'\bwater\s+restriction\b',
                r'\bdaily\s+fluid\s+restriction\b',
                r'\bon\s+fluid\s+restriction\b',
                r'\bmaintain\s+fluid\s+restriction\b',
                r'\blimit\s+fluid\s+intake\b',
                r'\blimited\s+fluid\s+intake\b',
                r'\besrd\s+on\s+hd\b',
                r'\besrd\s+on\s+dialysis\b',
                r'\bdialysis\b'
            ],
            text=t
        )

    def _heart_failure(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+(?:history\s+of\s+)?(?:heart|cardiac)\s+failure\b',
                r'\bden(?:ies|y|ied)\s+(?:heart|cardiac)\s+failure\b',
                r'\bwithout\s+(?:heart|cardiac)\s+failure\b',
                r'\bnegative\s+for\s+(?:heart|cardiac)\s+failure\b',
                r'\bno\s+(?:hf|chf|hfref|hfpef)\b'
            ],
            pos_patterns=[
                r'\bheart\s+failure\b',
                r'\bhf\b',
                r'\bchf\b',
                r'\bcongestive\s+heart\s+failure\b',
                r'\bcardiac\s+failure\b',
                r'\bhfref\b',
                r'\bhfpef\b',
                r'\bacute\s+on\s+chronic\s+heart\s+failure\b'
            ],
            text=t
        )

    def _renal_stone(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+(?:history\s+of\s+)?(?:renal|kidney)\s+stones?\b',
                r'\bden(?:ies|y|ied)\s+(?:renal|kidney)\s+stones?\b',
                r'\bwithout\s+(?:renal|kidney)\s+stones?\b',
                r'\bnegative\s+for\s+(?:renal|kidney)\s+stones?\b',
                r'\bno\s+(?:nephrolithiasis|urolithiasis|stone\s+disease)\b'
            ],
            pos_patterns=[
                r'\brenal\s+stones?\b',
                r'\bkidney\s+stones?\b',
                r'\bnephrolithiasis\b',
                r'\burolithiasis\b',
                r'\bcalculus\s+of\s+kidney\b',
                r'\bstone\s+disease\b',
                r'\bhistory\s+of\s+(?:renal|kidney)\s+stones?\b',
                r'\bhx\s+(?:of\s+)?(?:renal|kidney)\s+stones?\b',
                r'\bprior\s+(?:renal|kidney)\s+stones?\b',
                r'\bprevious\s+(?:renal|kidney)\s+stones?\b'
            ],
            text=t
        )

    def _hypercalcemia(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+hypercalcemia\b',
                r'\bno\s+hypercalcaemia\b',
                r'\bden(?:ies|y|ied)\s+hypercalcemia\b',
                r'\bden(?:ies|y|ied)\s+hypercalcaemia\b',
                r'\bcalcium\s+(?:level\s+)?(?:normal|within\s+normal\s+limits|wnl)\b',
                r'\bnormal\s+(?:serum\s+)?calcium\b'
            ],
            pos_patterns=[
                r'\bhypercalcemia\b',
                r'\bhypercalcaemia\b',
                r'\bhigh\s+(?:blood\s+)?(?:serum\s+)?calcium\s+level\b',
                r'\belevated\s+(?:serum\s+)?calcium\b',
                r'\bhigh\s+(?:serum\s+)?calcium\b'
            ],
            text=t
        )
    
    def _discharge_home(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bnot\s+discharg(?:ed)?\s+home\b',
                r'\bdischarged\s+to\s+(?:facility|rehab|nursing\s+home|snf)\b'
            ],
            pos_patterns=[
                r'\bdischarged\s+home\b',
                r'\bhome\s+discharge\b',
                r'\breturned\s+home\b'
            ],
            text=t
        )
    
    # ─── Tertiary variables ───────────────────────────────────────────────────
    def _delirium(self, t):
        return self._bool_neg_pos(
            [r'no\s+(?:signs?|evidence|features?)\s+of\s+delirium',
             r'den(?:ies|y)\s+delirium\b',
             r'(?:4at|dos|cam)\s*[:\-]?\s*(?:negative|0|nil)'],
            [r'\bdelirium\b',
             r'acute\s+(?:confusion|confusional\s+state)',
             r'(?:4at|dos|cam)\s*[:\-]?\s*positive',
             r'4at\s*[:\-]?\s*[4-9]\b',
             r'agitat(?:ed|ion)\b'], t)

    def _cognitive_impaired(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+(?:cognitive\s+)?(?:impairment|decline|deficit)\b',
                r'\bno\s+memory\s+loss\b',
                r'\bno\s+(?:history\s+of\s+)?dementia\b',
                r'\bno\s+(?:history\s+of\s+)?alzheimer\'?s?\s+disease\b',
                r'\bden(?:ies|y|ied)\s+(?:cognitive\s+impairment|memory\s+loss|dementia)\b',
                r'\b(?:normal|intact)\s+cognition\b',
                r'\bcognition\s+(?:normal|intact)\b',
                r'\bmoca\s*[:\-]?\s*(?:2[6-9]|30)\b',
                r'\bmmse\s*[:\-]?\s*(?:2[5-9]|30)\b'
            ],
            pos_patterns=[
                r'\bdementia\b',
                r'\bcognitive\s+impairment\b',
                r'\bcognitive\s+(?:decline|deficit)\b',
                r'\bmemory\s+loss\b',
                r'\bams\b',
                r'\baltered\s+mental\s+status\b',
                r'\balzheimer\'?s?\s+disease\b',
                r'\blewy\s+body\s+disease\b',
                r'\bmild\s+cognitive\s+impairment\b',
                r'\bmoca\s*[:\-]?\s*(?:[0-9]|1[0-9]|2[0-5])\b',
                r'\bmmse\s*[:\-]?\s*(?:[0-9]|1[0-9]|2[0-4])\b'
            ],
            text=t
        )

    def _syncope(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+(?:history\s+of\s+)?syncope\b',
                r'\bden(?:ies|y|ied)\s+syncope\b',
                r'\bwithout\s+syncope\b',
                r'\bnegative\s+for\s+syncope\b',
                r'\bno\s+(?:fainting|faint|blackout|black\s+out)\b',
                r'\bden(?:ies|y|ied)\s+(?:fainting|faint|blackout|black\s+out)\b',
                r'\bno\s+(?:loss\s+of\s+consciousness|loc)\b',
                r'\bden(?:ies|y|ied)\s+(?:loss\s+of\s+consciousness|loc)\b',
                r'\bremained\s+conscious\b',
                r'\bno\s+unexplained\s+fall\b'
            ],
            pos_patterns=[
                r'\bdizziness\b',
                r'\bdizzy\b',
                r'\blightheaded\b',
                r'\blightheadedness\b',
                r'\bsyncope\b',
                r'\bsyncopal\b',
                r'\bpresyncope\b',
                r'\bpre[-\s]?syncope\b',
                r'\bfaint\b',
                r'\bfainted\b',
                r'\bfainting\b',
                r'\bblackout\b',
                r'\bblack(?:ed)?\s+out\b',
                r'\bloss\s+of\s+consciousness\b',
                r'\bloc\b',
                r'\bunwitnessed\s+fall\b',
                r'\bunexplained\s+fall\b',
                r'\bvasovagal\b',
                r'\brecurrent\s+unexplained\s+falls?\b'
            ],
            text=t
        )

    def _orthostatic_hypo(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+(?:orthostatic|postural)\s+hypotension',
                r'(?:orthostatic|postural)\s+hypotension\s*:\s*(?:nil|no|absent|negative)'
            ],
            pos_patterns=[
                r'(?:orthostatic|postural)\s+hypotension\b',
                r'positional\s+drop\b',
             r'bp\s+drop\s+on\s+standing'], text=t)

    def _other_cardiac(self, t):
        return self._bool_neg_pos(
            [],
            [r'atrial\s+fibrillation\b|\baf\b',
             r'ventricular\s+(?:tachycardia|fibrillation)',
             r'heart\s+block\b',
             r'sick\s+sinus\b',
             r'pacemaker\b',
             r'icd\b'], t)

    def _eye_problem(self, t):
        return self._bool_neg_pos(
            [r'no\s+(?:eye|vision|visual)\s+(?:problem|issue|complaint)',
             r'(?:normal|intact|20/20|6/6)\s+vision'],
            [r'cataract\b',
             r'glaucoma\b',
             r'macular\s+degeneration\b',
             r'diabetic\s+retinopathy\b',
             r'blurred\s+vision\b',
             r'visual\s+(?:impairment|loss|deficit)',
             r'poor\s+(?:vision|sight)'], t)

    def _hearing_impaired(self, t):
        return self._bool_neg_pos(
            [r'no\s+(?:hearing|auditory)\s+(?:loss|impairment|problem)',
             r'normal\s+hearing\b'],
            [r'hearing\s+(?:loss|impairment|aid)\b',
             r'deaf(?:ness)?\b',
             r'wearing\s+(?:a\s+)?hearing\s+aid',
             r'hard\s+of\s+hearing\b'], t)

    def _urinary_incontinence(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+(?:urinary\s+)?incontinence\b',
                r'\bden(?:ies|y|ied)\s+(?:urinary\s+)?incontinence\b',
                r'\bwithout\s+(?:urinary\s+)?incontinence\b',
                r'\bcontinent\b',
                r'\bcontinent\s+of\s+urine\b',
                r'\bno\s+urine\s+leakage\b',
                r'\bno\s+leaking\s+urine\b',
                r'\bno\s+(?:uti|urinary\s+tract\s+infection|cystitis)\b',
                r'\bden(?:ies|y|ied)\s+(?:uti|urinary\s+tract\s+infection|cystitis)\b'
            ],
            pos_patterns=[
                r'\burinary\s+incontinence\b',
                r'\bincontinence\b',
                r'\bloss\s+of\s+bladder\s+control\b',
                r'\boveractive\s+bladder\b',
                r'\boab\b',
                r'\bneurogenic\s+bladder\b',
                r'\bbladder\s+dysfunction\b',
                r'\burinary\s+retention\b',
                r'\bbph\b',
                r'\bbenign\s+prostatic\s+hyperplasia\b',
                r'\buti\b',
                r'\burinary\s+tract\s+infection\b',
                r'\bcystitis\b',
                r'\bfrequent\s+urination\b',
                r'\bnocturia\b',
                r'\bnight\s+urination\b',
                r'\bdiaper\b',
                r'\burine\s+leakage\b',
                r'\baccidental\s+urination\b',
                r'\binvoluntary\s+urination\b',
                r'\bcannot\s+hold\s+urine\b',
                r'\bunable\s+to\s+hold\s+urine\b',
                r'\bwetting\s+episodes\b'
            ],
            text=t
        )

    def _fracture_risk(self, t):
        return self._bool_neg_pos(
            [r'no\s+(?:history\s+of\s+)?(?:fracture|osteoporosis)',
             r'normal\s+bone\s+density\b'],
            [r'osteoporosis\b',
             r'osteopenia\b',
             r'(?:previous|prior|history\s+of)\s+fracture\b',
             r'fragility\s+fracture\b',
             r'fracture\b',
             r'dexa\s+(?:scan\s+)?show(?:ing|s|ed)'], t)

    def _vertigo(self, t):
        # If BPPV is detected, this variable should be False
        if self._bppv(t) is True:
            return False

        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+vertigo\b',
                r'\bden(?:ies|y|ied)\s+vertigo\b',
                r'\bwithout\s+vertigo\b',
                r'\bnegative\s+for\s+vertigo\b',
                r'\bno\s+dizziness\b',
                r'\bden(?:ies|y|ied)\s+dizziness\b',
                r'\bno\s+spinning\s+sensation\b',
                r'\bden(?:ies|y|ied)\s+(?:room\s+spinning|spinning\s+sensation)\b'
            ],
            pos_patterns=[
                r'\bvertigo\b',
                r'\broom\s+spinning\b',
                r'\bspinning\s+sensation\b',
                r'\brotary\s+vertigo\b',
                r'\bpositional\s+vertigo\b',
                r'\bvertiginous\s+symptoms\b',
                r'\bdizziness\b'
            ],
            text=t
        )

    def _bppv(self, t):
        return self._bool_neg_pos(
            [r'(?:negative|normal)\s+dix[-\s]?hallpike\b'],
            [r'\bbppv\b',
             r'benign\s+paroxysmal\s+positional\s+vertigo\b',
             r'positive\s+dix[-\s]?hallpike\b'], t)

    def _vestibular_disease(self, t):
        return self._bool_neg_pos(
            [r'no\s+vestibular\s+(?:disease|dysfunction)',
             r'normal\s+(?:head\s+impulse|video\s+hit)\b'],
            [r'vestibular\s+(?:disease|dysfunction|neuritis|neuronitis)',
             r'labyrinthitis\b',
             r'positive\s+head\s+impulse\b'], t)

    def _foot_problem(self, t):
        return self._bool_neg_pos(
            [r'no\s+foot\s+(?:pain|problem|issue|complaint)'],
            [r'foot\s+(?:pain|problem|deformity|ulcer)',
             r'plantar\s+fasciitis\b',
             r'bunion\b',
             r'podiatrist\b',
             r'podiatry\b',
             r'neuropathic\s+foot\b',
             r'diabetic\s+foot\b'], t)

    def _non_terminal_pain(self, t):
        # If terminal/palliative pain is detected, classify non-terminal pain as False
        if self._terminal_pain(t) is True:
            return False

        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+(?:significant\s+)?pain\b',
                r'\bpain\s*:\s*(?:nil|none|no|0)\b',
                r'\bpain\s+free\b',
                r'\bden(?:ies|y|ied)\s+(?:any\s+)?pain\b',
                r'\bwithout\s+pain\b',
                r'\bno\s+(?:arthritis|joint|back|neck|musculoskeletal|bone|neuropathic|nerve)\s+pain\b'
            ],
            pos_patterns=[
                r'\bpain\b',
                r'\barthritis\s+pain\b',
                r'\bosteoarthritis\b',
                r'\bjoint\s+pain\b',
                r'\bback\s+pain\b',
                r'\blow\s+back\s+pain\b',
                r'\blower\s+back\s+pain\b',
                r'\bneck\s+pain\b',
                r'\bmusculoskeletal\s+pain\b',
                r'\bbone\s+pain\b',
                r'\bpost[-\s]?surgical\s+pain\b',
                r'\bpostoperative\s+pain\b',
                r'\btrauma\s+pain\b',
                r'\binjury\s+pain\b',
                r'\bneuropathic\s+pain\b',
                r'\bnerve\s+pain\b',
                r'\bneuralgia\b',
                r'\bdiabetic\s+neuropathy\b',
                r'\bperipheral\s+neuropathy\b',
                r'\bburning\s+pain\b',
                r'\bshooting\s+pain\b',
                r'\btingling\s+pain\b',
                r'\bpain\s+(?:score|scale|rating|level)\s*[:\-]?\s*[1-9]\b',
                r'\bsevere\s+pain\b',
                r'\bmoderate\s+pain\b',
                r'\bmild\s+pain\b',
                r'\bchronic\s+pain\b',
                r'\bsignificant\s+pain\b'
            ],
            text=t
        )

    def _terminal_pain(self, t):
        return self._bool_neg_pos(
            [],
            [r'(?:cancer|carcinoma|malignancy|oncology).*pain',
             r'pain.*(?:cancer|carcinoma|malignancy)',
             r'palliative\s+care\b',
             r'tumour\s+pain\b',
             r'(?:terminal|end[-\s]of[-\s]life)\s+(?:illness|disease|care)',
             r'(?:copd|cardiac\s+failure|heart\s+failure).*pain'], t)

    def _malnutrition(self, t):
        return self._bool_neg_pos(
            [r'no\s+(?:malnutrition|nutritional\s+deficit)',
             r'adequate\s+(?:nutrition|diet|oral\s+intake)',
             r'(?:normal|healthy)\s+(?:bmi|weight)'],
            [r'malnutrition\b',
             r'malnourished\b',
             r'unintentional\s+weight\s+loss\b',
             r'loss\s+of\s+appetite\b',
             r'sarcopenia\b',
             r'underweight\b'], t)

    def _walk_with_aid(self, t):
        return self._bool_neg_pos(
            [r'walks?\s+(?:independently|without\s+aid|unaided)'],
            [r'walks?\s+with\s+(?:a\s+)?(?:stick|cane|frame|walker|rollator|zimmer|aid)',
             r'(?:uses?|requires?)\s+(?:a\s+)?(?:walking\s+)?(?:stick|cane|frame|walker|rollator)',
             r'mobility\s+aid\b',
             r'wheelchair\b',
             r'chair\s+or\s+wheelchair\b'], t)

    def _fear_of_falling(self, t):
        return self._bool_neg_pos(
            [r'no\s+fear\s+of\s+fall(?:ing)?',
             r'den(?:ies|y)\s+fear\s+of\s+fall(?:ing)?',
             r'not\s+afraid\s+of\s+fall(?:ing)?'],
            [r'fear\s+of\s+fall(?:ing)?',
             r'worried\s+about\s+fall(?:ing)?',
             r'anxious\s+about\s+fall(?:ing)?',
             r'concerned\s+about\s+fall(?:ing)?'], t)

    #FRIDs medications
    BASE_DIR = Path(__file__).resolve().parent
    FRIDS_PATH = BASE_DIR / "frids_library.json"

    with FRIDS_PATH.open("r", encoding="utf-8") as f:
        FRIDS_LIBRARY = json.load(f)

    def _load_frids_medication_names(frids_library: dict) -> list[str]:
        # Otherwise extract names from categories
        medication_names = []

        for category_data in frids_library.get("categories", {}).values():
            for med in category_data.get("medications", []):
                name = med.get("name")
                if name:
                    medication_names.append(name.strip().lower())

        return sorted(set(medication_names))


    FRIDS_MEDICATION_NAMES = _load_frids_medication_names(FRIDS_LIBRARY)

    FRIDS_PATTERNS = [
        re.compile(rf"\b{re.escape(med)}\b", re.IGNORECASE)
        for med in FRIDS_MEDICATION_NAMES
    ]
    
    def _gait_balance(self, t):
        # If unsteady_gait is detected, classify poor gait/balance/muscle strength as True
        if self._unsteady_gait(t) is True:
            return True

        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+(?:muscle\s+)?weakness\b',
                r'\bden(?:ies|y|ied)\s+(?:muscle\s+)?weakness\b',
                r'\bwithout\s+(?:muscle\s+)?weakness\b',
                r'\bnormal\s+(?:muscle\s+)?strength\b',
                r'\bstrength\s+(?:is\s+)?(?:normal|intact|preserved)\b',
                r'\b5/5\s+(?:strength|power)\b',
                r'\bfull\s+(?:muscle\s+)?strength\b',
                r'\bnot\s+frail\b',
                r'\bno\s+frailty\b',
                r'\bno\s+deconditioning\b',
                r'\bnot\s+deconditioned\b',
                r'\bno\s+sarcopenia\b',
                r'\bromberg\s+(?:test\s+)?(?:negative|absent)\b',
                r'\bnegative\s+romberg\b',
                r'\bno\s+romberg\b'
            ],
            pos_patterns=[
                r'\bmuscle\s+weakness\b',
                r'\bweakness\b',
                r'\bdeconditioning\b',
                r'\bdeconditioned\b',
                r'\bsarcopenia\b',
                r'\bsarcopenic\b',
                r'\breduced\s+(?:muscle\s+)?strength\b',
                r'\bfrailty\b',
                r'\bfrail\b',
                r'\bstage\s*[12]\s+balance\s+test\b',
                r'\bone\s+leg\s+stand\s*(?:<|less\s+than)\s*10\s*(?:s|sec|seconds)\b',
                r'\bberg\s+balance\s+scale\s*(?:<|less\s+than)\s*41\b',
                r'\btinetti\s+test\s*(?:<|less\s+than)\s*24\b',
                r'\bpoma\s*(?:<|less\s+than)\s*24\b',
                r'\bmini[-\s]?best\s+test\s*(?:<|less\s+than)\s*18\.?5\b',
                r'\bmrc\s+scale\s*(?:<|less\s+than)\s*48\b',
                r'\bromberg\s+(?:test\s+)?(?:present|positive)\b',
                r'\bpositive\s+romberg\b'
            ],
            text=t
        )

    
    def _functional_disabled(self, t):
        return self._bool_neg_pos(
            [r'activity\s+status\s*:\s*ambulatory\s*-\s*independent',
             r'walks?\s+independently',
             r'independent\s+with\s+adl'],
            [r'out\s+of\s+bed\s+with\s+assistance',
             r'assistance\s+to\s+chair\s+or\s+wheelchair',
             r'needs?\s+assistance\s+with\s+(?:mobility|transfers|ambulation|adl)',
             r'requires?\s+(?:one|1|two|2|assist|assistance)',
             r'extended\s+care',
             r'discharge\s+disposition\s*:\s*extended\s+care',
             r'wheelchair\s+bound',
             r'assisted\s+living',
             r'snf\b',
             r'skilled\s+nursing'], t)
    
    def _vit_d_deficiency(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+vitamin\s+d\s+(?:deficiency|deficit|insufficiency)\b',
                r'\bden(?:ies|y|ied)\s+vitamin\s+d\s+(?:deficiency|deficit|insufficiency)\b',
                r'\bwithout\s+vitamin\s+d\s+(?:deficiency|deficit|insufficiency)\b',
                r'\bnormal\s+vitamin\s+d\b',
                r'\bsufficient\s+vitamin\s+d\b'
            ],
            pos_patterns=[
                r'\bvitamin\s+d\s+(?:deficiency|deficit|insufficiency)\b',
                r'\blow\s+vitamin\s+d\b',
                r'\binadequate\s+vitamin\s+d\b',
                r'\bvdd\b'
            ],
            text=t
        )

    def _cal_deficiency(self, t):
        return self._bool_neg_pos(
            neg_patterns=[
                r'\bno\s+calcium\s+(?:deficiency|deficit|insufficiency)\b',
                r'\bden(?:ies|y|ied)\s+calcium\s+(?:deficiency|deficit|insufficiency)\b',
                r'\bwithout\s+calcium\s+(?:deficiency|deficit|insufficiency)\b',
                r'\bnormal\s+calcium\b',
                r'\bsufficient\s+calcium\b'
            ],
            pos_patterns=[
                r'\bcalcium\s+(?:deficiency|deficit|insufficiency)\b',
                r'\blow\s+calcium\b',
                r'\binadequate\s+calcium\b'
            ],
            text=t
        )
# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RuleEngine:
    """
    Applies Evidence Library rules to PatientVariables and returns a DecisionOutput.
    Rules are organised by:
      1. General recommendations (all risk levels)
      2. Low Risk path
      3. Intermediate Risk path
      4. High Risk path (multifactorial)
    """

    def evaluate(self, v: PatientVariables) -> DecisionOutput:
        warnings = self._check_missing_core(v)
        general = self._general_recommendations(v, warnings)

        p3kq = v.pass_3kq
        pfs  = v.pass_fall_severity

        # ── Low Risk: 3KQ all negative ────────────────────────────────────────
        if p3kq is True:
            recs, w = self._low_risk_exercise(v)
            return DecisionOutput(
                risk_level="Low Risk",
                condition="Fall Screening (3 Key Questions) negative",
                recommendations=general + recs,
                extracted_variables=self._serialise(v),
                warnings=warnings + w,
            )

        # ── Requires fall screen positive ─────────────────────────────────────
        if p3kq is False:
            # Path B: non-severe fall + steady gait → same low-risk recommendations
            if pfs is True and v.unsteady_gait is False:
                recs, w = self._low_risk_exercise(v)
                return DecisionOutput(
                    risk_level="Low Risk",
                    condition=(
                        "Fall Screening positive, non-severe fall, steady gait "
                        "→ Low Risk pathway"
                    ),
                    recommendations=general + recs,
                    extracted_variables=self._serialise(v),
                    warnings=warnings + w,
                )

            # Intermediate Risk: non-severe fall + unsteady/poor gait
            if (pfs is True and
                    (v.unsteady_gait is True or
                     v.poor_gait_balance_muscle_strength is True)):
                recs, w = self._intermediate_risk_rules(v)
                return DecisionOutput(
                    risk_level="Intermediate Risk",
                    condition=(
                        "Fall Screening positive, non-severe fall, "
                        "unsteady gait or poor balance/muscle strength"
                    ),
                    recommendations=general + recs,
                    extracted_variables=self._serialise(v),
                    warnings=warnings + w,
                )

            # High Risk: severe fall or fall severity not passed
            if pfs is False:
                recs, w = self._high_risk_rules(v)
                return DecisionOutput(
                    risk_level="High Risk",
                    condition=(
                        "Fall Screening positive with severe fall indicators "
                        "→ Multifactorial assessment"
                    ),
                    recommendations=general + recs,
                    extracted_variables=self._serialise(v),
                    warnings=warnings + w,
                )

            
            # Fall screen positive but fall severity unknown
            return DecisionOutput(
                risk_level="Intermediate / High Risk (pending severity assessment)",
                condition=(
                    "Fall Screening positive – fall severity assessment required "
                    "to determine Intermediate vs High Risk pathway"
                ),
                recommendations=general + [Recommendation(
                    category="Assessment Required",
                    text=(
                        "Complete Fall Severity Assessment (unable to get up, number of falls in a year, "
                        "loss of consciousness, fall with injury, frailty status) "
                        "to determine the appropriate risk pathway."
                    ),
                    source="System",
                )],
                extracted_variables=self._serialise(v),
                warnings=warnings + [
                    "Fall severity assessment variables could not be extracted. "
                    "Clinician must complete severity assessment to route correctly."
                ],
            )
                

        # ── Indeterminate: 3KQ result cannot be determined ────────────────────
        if self._should_infer_high_risk(v):
            recs, w = self._high_risk_rules(v)
            return DecisionOutput(
                risk_level="High Risk (inferred from documented risk factors)",
                condition=(
                    "Fall screening questions not explicitly documented, but fall "
                    "severity indicators suggest high risk"
                ),
                recommendations=general + recs,
                extracted_variables=self._serialise(v),
                warnings=warnings + w + [
                    "High Risk pathway inferred from documented fall severity because "
                    "3 Key Questions were not explicitly available in the text."
                ],
            )

        if self._has_mobility_concern(v):
            recs, w = self._intermediate_risk_rules(v)
            return DecisionOutput(
                risk_level="Intermediate Risk (inferred from mobility concerns)",
                condition=(
                    "Fall screening questions not explicitly documented, but mobility "
                    "concerns were identified in the discharge summary"
                ),
                recommendations=general + recs,
                extracted_variables=self._serialise(v),
                warnings=warnings + w + [
                    "Intermediate Risk pathway inferred from documented mobility concerns "
                    "because 3 Key Questions were not explicitly available in the text."
                ],
            )

        return DecisionOutput(
            risk_level="Indeterminate",
            condition="Fall screening result could not be determined from text",
            recommendations=general + [Recommendation(
                category="Assessment Required",
                text=(
                    "Unable to determine fall risk level automatically. "
                    "Please manually assess the 3 Key Questions:\n"
                    "  Q1: Have you had a fall in the past year?\n"
                    "  Q2: Do you feel unsteady when standing or walking?\n"
                    "  Q3: Do you worry about falling?"
                ),
                source="System",
            )],
            extracted_variables=self._serialise(v),
            warnings=warnings,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GENERAL RECOMMENDATIONS (all risk levels)
    # ─────────────────────────────────────────────────────────────────────────
    def _general_recommendations(self, v: PatientVariables, warnings: list) -> list:
        recs = []

        recs.append(Recommendation(
            category="Medication",
            text=(
                "a. Review medications regularly, including after any major health event "
                "or hospital discharge (care transition) and before prescribing a new medicine.\n"
                "b. Tell your doctor about your current medications. Make a list of every "
                "medicine you take, include ones that could be discontinued, and show it to "
                "your doctor during your next appointment. Include herbs, tablets or supplements "
                "from health food stores, supermarkets, and pharmacies.\n"
                "c. Monitor any adverse effects, drug interactions and toxicity of medicines."
            ),
            source="ACSQHC",
        ))

        recs.append(Recommendation(
            category="Environment",
            text=(
                "a. Have good lighting, especially between the bed and the bathroom at night.\n"
                "b. Remove clutter and keep walkways and corridors clear and well lit.\n"
                "c. Check that mats and rugs are secure with no tears or wrinkles.\n"
                "d. Install grab rails in the bathroom.\n"
                "e. Remove mosses that make garden pathways slippery when wet.\n"
                "f. Repair broken, uneven or cracked paths, patios and other walking surfaces. "
                "Use non-slip tape or paint on the leading edge of outdoor steps."
            ),
            source="ACSQHC",
        ))

        recs.append(Recommendation(
            category="Vision",
            text=(
                "a. Have annual eye examinations with an optometrist to maximise vision.\n"
                "b. Ensure accessible, clean glasses are worn. If the person uses different "
                "glasses for reading and distance, encourage distance glasses when mobilising.\n"
                "c. Vision impairment is not part of normal ageing — manage risk factors "
                "(diabetes, hypertension, steroid use).\n"
                "d. Limit exposure to ultraviolet light by wearing sunglasses and hats.\n"
                "e. Cease smoking, as it increases risk of eye diseases (cataracts, macular degeneration)."
            ),
            source="ACSQHC, ICOPE",
        ))

        recs.append(Recommendation(
            category="Hearing",
            text=(
                "a. If wearing a hearing aid, encourage use when mobilising and ensure it is working.\n"
                "b. Consult audiologist for detailed assessment and fall-specific hearing examination; "
                "annual hearing assessment is encouraged.\n"
                "c. Clean only the outer part of the ear with a soft cloth. "
                "Do not insert objects into ears.\n"
                "d. Avoid loud sounds and use earplugs in noisy places."
            ),
            source="ACSQHC, ICOPE",
        ))

        recs.append(Recommendation(
            category="Foot Problem and Footwear",
            text=(
                "a. Use safe, well-fitting footwear: low square heels for stability, "
                "supporting ankle collar, tread soles to prevent slips, firm soles to "
                "optimise foot position sense.\n"
                "b. Wear enclosed sturdy shoes around the house rather than slip-on footwear."
            ),
            source="World Guideline, ICOPE",
        ))

        recs.append(Recommendation(
            category="Nutrition",
            text=(
                "a. Eat smaller amounts more often (5–6 times per day) if large meals are difficult.\n"
                "b. Carbohydrate intake should primarily come from wholegrains, vegetables, fruits "
                "and pulses; fats from wholefoods such as nuts, seeds, beans, olives and fatty fish.\n"
                "c. Include vitamin D and calcium-rich foods for bone health and fracture prevention. "
                "Safe sunlight exposure supports vitamin D synthesis.\n"
                "d. Decrease caffeine and alcohol, particularly at night, which can increase urinary urgency."
            ),
            source="ICOPE",
        ))

        # Hydration — excluded if fluid_restriction OR heart_failure
        if v.fluid_restriction is True or v.heart_failure is True:
            warnings.append(
                "General hydration recommendation NOT applied — "
                "patient has fluid restriction or heart failure."
            )
        else:
            note = ""
            if v.fluid_restriction is None or v.heart_failure is None:
                note = " (clinician to verify no fluid restriction or heart failure)"
                warnings.append(
                    "Hydration recommendation applied but fluid restriction / heart failure "
                    "status could not be confirmed from text — clinician to verify."
                )
            recs.append(Recommendation(
                category="Nutrition",
                text=(
                    "Hydration: drink at least 1.6–2 L (6–8 glasses or cups) per day, "
                    "regularly throughout the day. Increase intake during hot weather. "
                    "Additional fluids will come from food." + note
                ),
                source="ICOPE",
                is_conditional=True,
                condition_note="Excluded if fluid restriction or heart failure.",
            ))

        recs.append(Recommendation(
            category="Fracture Risk Assessment",
            text=(
                "Access bone mineral densitometry assessment / Dual Energy X-Ray (DEXA) "
                "scan to identify osteoporosis."
            ),
            source="ACSQHC",
        ))

        recs.append(Recommendation(
            category="Fall Emergency Plan",
            text=(
                "a. Falls action plan:\n"
                "   • Know who to call for help. Keep a list of phone numbers near the phone.\n"
                "   • Have a phone within reach on a low table in case it is hard to get back up.\n"
                "   • Let trusted family/friends know how to access the home in case of a fall.\n"
                "   • Consider a personal alarm device.\n"
                "b. Steps after a fall:\n"
                "   • Don't panic. Stay still for a few minutes, stay calm.\n"
                "   • Ensure you are not injured before trying to move.\n"
                "   • To get up: bend knees, roll onto side, crawl to stable furniture, "
                "bring one knee forward, push up with arms.\n"
                "   • If unable to get up: keep warm, use personal alarm or phone, "
                "bang an object to alert neighbours. Dial 000 for emergency services.\n"
                "   • Always tell your doctor if you have had a fall."
            ),
            source="NSW Fall Prevention Module",
        ))

        return recs

    # ─────────────────────────────────────────────────────────────────────────
    # LOW RISK
    # ─────────────────────────────────────────────────────────────────────────
    def _low_risk_exercise(self, v: PatientVariables):
        recs = []
        warnings = []

        recs.append(Recommendation(
            category="Exercise",
            text=(
                "a. At least 150–300 minutes of moderate-intensity aerobic physical activity, "
                "or at least 75–150 minutes of vigorous-intensity aerobic activity, "
                "or an equivalent combination throughout the week.\n"
                "b. Muscle-strengthening activities at moderate or greater intensity involving "
                "all major muscle groups on 2 or more days a week.\n"
                "c. The WHO and UK Physical Activity guidelines recommend activities that "
                "challenge balance and include resistance training twice per week. "
                "General walking alone is unlikely to prevent falls.\n"
                "d. Varied multicomponent physical activity emphasising functional balance and "
                "strength training at moderate or greater intensity, on 3 or more days a week.\n"
                "e. Attend community fall prevention exercise programmes such as tai chi and "
                "strength/balance training: https://www.activeandhealthy.nsw.gov.au"
            ),
            source="WHO Sedentary Guidelines",
        ))

        # Specific exercise recommendation depends on APSS result
        if v.pass_apss is False:
            # APSS flagged — seek professional guidance before exercise
            recs.append(Recommendation(
                category="Exercise",
                text=(
                    "Please seek guidance from an appropriate allied health professional "
                    "or medical practitioner prior to undertaking fall prevention exercise. "
                    "Balance-challenging and functional exercises (e.g. sit-to-stand, stepping) "
                    "should be offered with sessions three or more times weekly, individualised "
                    "and progressed in intensity for at least 12 weeks. "
                    "Professionals include physiotherapists, exercise physiologists or "
                    "kinesiologists, trained exercise instructors, or other allied health professionals."
                ),
                source="ACSQHC",
                grade="1A",
            ))
        elif v.pass_apss is True:
            # APSS passed — light-to-moderate exercise programme
            recs.append(Recommendation(
                category="Exercise",
                text=(
                    "Light to moderate intensity exercise is recommended for at least 12 weeks. "
                    "Increase volume and intensity slowly. Discuss any progression (volume, "
                    "intensity, duration, modality) with an exercise professional to optimise results.\n"
                    "NSW Active and Healthy exercise circuit: "
                    "https://www.activeandhealthy.nsw.gov.au/active-living/fact-sheets-and-physical-activity-manual/"
                    "healthy-ageing-resources/exercise-circuit-1"
                ),
                source="ACSQHC",
                grade="1A",
            ))
        else:
            warnings.append(
                "APSS result could not be determined — specific exercise intensity "
                "recommendation could not be applied. Clinician to verify APSS status."
            )

        return recs, warnings

    # ─────────────────────────────────────────────────────────────────────────
    # INTERMEDIATE RISK
    # ─────────────────────────────────────────────────────────────────────────
    def _intermediate_risk_rules(self, _v: PatientVariables):
        recs = []
        warnings = []

        recs.append(Recommendation(
            category="Exercise",
            text=(
                "a. Undertake 2 to 3 hours of exercise per week on an ongoing basis to prevent falls.\n"
                "b. Tailored exercises on balance, gait and strength, delivered by appropriately "
                "trained professionals who can adapt exercises to functional status and co-morbidities "
                "(physiotherapists, exercise physiologists, kinesiologists, trained exercise instructors).\n"
                "c. Fall prevention exercises should focus on maintaining balance during functional "
                "tasks needed for daily life. Types include strength/resistance training, "
                "aerobic/cardiovascular training, balance training and flexibility training.\n"
                "d. Effective programmes include individualised exercises supporting daily tasks: "
                "sit-to-stand, squats, reaching while standing, stepping and walking in different "
                "directions, speeds, environments and while dual-tasking. Weights can be added.\n"
                "e. Exercises should be challenging but safe and achievable. "
                "Review and progress regularly to maintain optimal difficulty."
            ),
            source="World Guideline, ACSQHC",
            grade="1B",
        ))

        return recs, warnings

    # ─────────────────────────────────────────────────────────────────────────
    # HIGH RISK (multifactorial)
    # ─────────────────────────────────────────────────────────────────────────
    def _high_risk_rules(self, v: PatientVariables):
        recs = []
        warnings = []

        # Medication (FRIDs)
        if v.pass_FRIDs is False:
            frids_parts = []
            for category, drugs in v.detected_FRIDs.items():
                frids_parts.append(f"{category}: {', '.join(drugs)}")
            frids_text = "; ".join(frids_parts) if frids_parts else "fall-risk-increasing drugs"

            recs.append(Recommendation(
                category="Medication",
                text=(
                    f"FRIDs detected: {frids_text}. "
                    "Consult pharmacist for medication review advice and seek clinician for "
                    "medication review when necessary, in partnership with the patient to report "
                    "symptoms and education strategies to minimise fall risk."
                ),
                source="World Guideline",
                grade="1B",
            ))
        elif v.pass_FRIDs is None:
            warnings.append(
                "FRIDs status could not be confirmed from text — clinician to review "
                "medications for fall-risk-increasing drugs (benzodiazepines, antidepressants, "
                "antipsychotics, opioids, antiepileptics, diuretics, vasodilators, alpha-blockers, "
                "sedative antihistamines, overactive bladder medications)."
            )

        # Fear of falling
        if v.concerns_about_falling is True:
            recs.append(Recommendation(
                category="Concerns About Falling",
                text=(
                    "a. Supervised holistic exercise interventions in community settings "
                    "(e.g. Pilates or yoga).\n"
                    "b. Cognitive behavioural therapy.\n"
                    "c. Occupational therapy."
                ),
                source="World Guideline",
                grade="1B",
            ))

        # Cognition — delirium
        if v.is_delirium is True:
            recs.append(Recommendation(
                category="Cognitive",
                text=(
                    "a. Consult an occupational therapist for adapting the environment to promote "
                    "safety and educating caregivers in strategies for safe mobility.\n"
                    "b. Consult GP to identify and treat underlying causes of delirium.\n"
                    "c. Family reassurance and support."
                ),
                source="World Guideline",
            ))

        # Cognition — impaired (but not delirium)
        if v.is_cognitive_impaired is True:
            recs.append(Recommendation(
                category="Cognitive",
                text=(
                    "a. Identify and modify environmental fall risk factors.\n"
                    "b. Modify lifestyle in terms of diet/nutrition and exercise routines "
                    "to reduce fall risks; maintain detailed recording of fall incidents.\n"
                    "c. Consult specialist for cognitive decline management (clinician for "
                    "medication and associated disease management; psychologist/OT for cognitive training).\n"
                    "d. Consider finding daily activity support / carer.\n"
                    "e. Provide WHO dementia support resources."
                ),
                source="NSW Fall Prevention Module, World Guidelines",
            ))

        # Cognition — normal (rule out impairment)
        if v.is_delirium is False and v.is_cognitive_impaired is False:
            recs.append(Recommendation(
                category="Cognitive",
                text="Consult medical professionals for cognitive assessment.",
                source="World Guideline",
                grade="1B",
            ))

        # Cardiovascular — syncope
        if v.have_syncope is True:
            recs.append(Recommendation(
                category="Cardiovascular",
                text=(
                    "a. Education/lifestyle measures: awareness and avoidance of triggers.\n"
                    "b. Review hypotensive medication.\n"
                    "c. Consult clinician for pharmacological therapy or cardiac pacing."
                ),
                source="Martone et al. (2024). J Clin Med, 13(3), 727.",
            ))

        # Cardiovascular — orthostatic hypotension
        if v.have_orthostatic_hypo is True:
            text_parts = [
                "a. Education/lifestyle measures: awareness and avoidance of triggers and situations.",
                "b. Sufficient salt and water intake (fluid 2–3 L/day and 10 g sodium chloride) — "
                "if not contraindicated.",
                "c. Rapid ingestion of cool water.",
                "d. Review vasoactive medication.",
                "e. Consult clinician for pharmacological therapy or cardiac pacing.",
                "f. Abdominal binders and compression stockings (fit with OT and physiotherapist).",
            ]
            if v.fluid_restriction is True or v.heart_failure is True:
                text_parts[1] = (
                    "b. Salt and water intake recommendation NOT applied — "
                    "fluid restriction or heart failure present."
                )
                warnings.append(
                    "Orthostatic hypotension: salt/water intake recommendation excluded "
                    "due to fluid restriction or heart failure."
                )
            recs.append(Recommendation(
                category="Cardiovascular",
                text="\n".join(text_parts),
                source="Martone et al. (2024). J Clin Med, 13(3), 727.",
            ))

        # Cardiovascular — other cardiac condition
        if v.other_cardiac_condition is True:
            recs.append(Recommendation(
                category="Cardiovascular",
                text=(
                    "Consult clinician for further assessment and appropriate medical management "
                    "(pharmacological therapy, ICD implantation, or surgical intervention)."
                ),
                source="Martone et al. (2024). J Clin Med, 13(3), 727.",
            ))

        # Pain — non-terminal
        if v.non_terminal_pain is True:
            recs.append(Recommendation(
                category="Pain",
                text=(
                    "a. Both non-pharmacological (physiotherapy, cognitive behavioural therapy) "
                    "and pharmacological approaches need to be considered.\n"
                    "b. For all analgesics: start slow, go slow, and monitor efficacy and adverse effects."
                ),
                source="World Guidelines",
                grade="E",
            ))

        # Pain — terminal
        if v.terminal_pain is True:
            recs.append(Recommendation(
                category="Pain",
                text=(
                    "For pain related to terminal/chronic and progressive/critical illnesses "
                    "(e.g. cancer, COPD, cardiac failure), referral to palliative team for "
                    "multidomain management is recommended."
                ),
                source="ICOPE",
            ))

        # Foot problem
        if v.with_foot_problem is True:
            recs.append(Recommendation(
                category="Foot Problem and Footwear",
                text="Consult podiatrist for foot pain and foot problems.",
                source="World Guidelines",
            ))

        # Sensory — vertigo
        if v.have_vertigo is True:
            recs.append(Recommendation(
                category="Dizziness and Vertigo",
                text=(
                    "a. Consult clinician and undertake follow-up assessment to identify "
                    "cardiovascular, neurological and/or vestibular causes of vertigo.\n"
                    "b. Review medicines regimen to identify any medicines contributing to "
                    "dizziness or postural hypotension (antihypertensives, antidepressants, "
                    "anticholinergics, hypoglycaemics)."
                ),
                source="World Guidelines, ACSQHC",
                grade="E",
            ))

        # Sensory — BPPV
        if v.have_BPPV is True:
            recs.append(Recommendation(
                category="Dizziness and Vertigo",
                text="Consult clinician for intervention such as particle repositioning manoeuvres.",
                source="World Guidelines, ACSQHC",
                grade="E",
            ))

        # Sensory — vestibular disease
        if v.have_vestibular_disease is True:
            recs.append(Recommendation(
                category="Dizziness and Vertigo",
                text=(
                    "Referral to OT/physiotherapist for vestibular rehabilitation therapy (VRT) "
                    "to improve postural and gait stability."
                ),
                source="World Guidelines, ACSQHC",
                grade="E",
            ))

        # Sensory — vision impaired
        if v.have_eye_problem is True:
            recs.append(Recommendation(
                category="Vision",
                text=(
                    "a. Comprehensive eye and vision examination by a health professional "
                    "with specialised knowledge (ophthalmic nurse/optometrist/ophthalmologist).\n"
                    "b. Systematic retinal check (annually or biennially) for people with "
                    "hypertension or diabetes.\n"
                    "c. Cataract surgery: For older people with clinically significant visual "
                    "impairment primarily due to cataracts, facilitate timely referral for cataract "
                    "surgery in both eyes (unless contraindicated). (Level 1A)\n"
                    "d. Eyewear prescription: Advise active older people to use single-lens distance "
                    "glasses rather than bifocal/multifocal/progressive lenses when active outdoors. "
                    "(Level 2B)\n"
                    "e. Occupational therapy interventions involving home hazard reductions "
                    "(larger type/good contrast household objects, talking assistive products)."
                ),
                source="ICOPE, ACSQHC, World Guidelines",
                grade="1A, 2B",
            ))

        # Sensory — hearing impaired
        if v.hearing_impaired is True:
            recs.append(Recommendation(
                category="Hearing",
                text=(
                    "Seek audiologist for comprehensive assessment and advice on the necessity "
                    "and fitting of a hearing aid."
                ),
                source="ICOPE",
            ))

        # Urinary incontinence
        if v.urinary_incontinence is True:
            recs.append(Recommendation(
                category="Urinary Symptoms and Incontinence",
                text=(
                    "a. Toilet access: referral to OT for home modifications for easier access. "
                    "Consider assistive products (raised toilet seat, commode, chair).\n"
                    "b. Consider using containment products.\n"
                    "c. Social care and support: provide assistance with toileting, bathing, "
                    "dressing, hygiene and sanitation products.\n"
                    "d. Review medications to identify any that may lead to urinary incontinence."
                ),
                source="ICOPE",
            ))

        # Environment modification
        if v.home_modification_need is True:
            recs.append(Recommendation(
                category="Environment Modifications",
                text=(
                    "a. Referral to occupational therapist for living environment assessment, "
                    "tailored home safety interventions and education about safety strategies, "
                    "equipment, aids and devices.\n"
                    "b. OT assessment for appropriateness and proper use of walking aids."
                ),
                source="ACSQHC",
            ))

        # Functional disability / walking aid
        if v.functional_disabled is True or v.walk_with_aid is True:
            recs.append(Recommendation(
                category="Functional Ability and Walking Aids",
                text=(
                    "a. Refer to occupational therapist for comprehensive daily living function "
                    "assessment and interventions.\n"
                    "b. Refer to social worker for community/social support."
                ),
                source="ICOPE",
            ))

        # Nutrition — malnutrition
        if v.mal_nutrition is True:
            recs.append(Recommendation(
                category="Nutrition",
                text="Consult dietitian for diet modification.",
                source="ACSQHC",
            ))

        # Nutrition — calcium deficiency
        if v.cal_deficiency is True:
            recs.append(Recommendation(
                category="Nutrition",
                text=(
                    "a. Choose high-calcium foods and exclude foods that limit calcium absorption.\n"
                    "b. Review medicines regimen when commencing calcium supplements, as certain "
                    "medicines may interact or adversely affect calcium levels."
                ),
                source="ACSQHC",
            ))

        # Nutrition — vitamin D deficiency
        if v.vit_D_deficiency is True:
            recs.append(Recommendation(
                category="Nutrition",
                text=(
                    "Vitamin D supplements — dosage as suggested by clinician/dietitian "
                    "(max dose 500–600 mg/day)."
                ),
                source="ACSQHC",
            ))

        # Fracture risk
        if v.with_fracture_risk is True:
            recs.append(Recommendation(
                category="Fracture Risk",
                text=(
                    "a. Improve muscle strength, optimise functional capacity and improve "
                    "safety of the older person's environment.\n"
                    "b. Lifestyle measures: regular weight-bearing exercises, balanced diet "
                    "including adequate calcium, avoid smoking and excessive alcohol.\n"
                    "c. Consult clinician for pharmacological therapy for osteoporosis "
                    "(e.g. oral bisphosphonates). For those with difficulty following medication "
                    "instructions, consider long-acting injectable alternatives.\n"
                    "d. Co-prescribe vitamin D with calcium for bone health, as directed by "
                    "a medical practitioner.\n"
                    "e. Consider hip protectors to reduce fall-related fracture risk. "
                    "Consult health professionals for correct use and regular checks of fit, "
                    "position, skin integrity, toileting independence and comfort."
                ),
                source="ACSQHC",
            ))

        if not recs:
            warnings.append(
                "High Risk pathway entered but no specific multifactorial conditions were "
                "identified from the text. Clinician to complete full multifactorial assessment."
            )

        return recs, warnings

    # ─── Helpers ──────────────────────────────────────────────────────────────
    def _check_missing_core(self, v: PatientVariables) -> list:
        warnings = []
        for attr, label in [
            ("fall_within_one_year", "Q1 – fall in the past year (count)"),
            ("feel_unsteady",        "Q2 – feels unsteady"),
            ("worry_fall",           "Q3 – worries about falling"),
        ]:
            if getattr(v, attr) is None:
                warnings.append(f"Could not extract from text: {label}")
        return warnings

    def _should_infer_high_risk(self, v: PatientVariables) -> bool:
        return v.pass_fall_severity is False

    def _has_mobility_concern(self, v: PatientVariables) -> bool:
        return any([
            v.functional_disabled is True,
            v.walk_with_aid is True,
            v.unsteady_gait is True,
            v.poor_gait_balance_muscle_strength is True,
        ])

    def _serialise(self, v: PatientVariables) -> dict:
        d = {
            k: val for k, val in v.__dict__.items()
            if k not in ("extraction_notes", "evidence_status", "evidence_notes")
        }
        evidence_status = dict(v.evidence_status)
        evidence_notes = dict(v.evidence_notes)
        d["pass_3kq"] = v.pass_3kq
        d["pass_fall_severity"] = v.pass_fall_severity
        evidence_status.setdefault(
            "pass_3kq",
            "derived_positive" if v.pass_3kq is True else
            "derived_negative" if v.pass_3kq is False else
            "not_documented_or_not_extractable",
        )
        evidence_status.setdefault(
            "pass_fall_severity",
            "derived_positive" if v.pass_fall_severity is True else
            "derived_negative" if v.pass_fall_severity is False else
            "not_documented_or_not_extractable",
        )
        evidence_notes.setdefault(
            "pass_3kq",
            "Derived from fall history, unsteadiness, and worry-about-falling variables.",
        )
        evidence_notes.setdefault(
            "pass_fall_severity",
            "Derived from fall count severity, inability to get up, LOC/syncope, injury, and frailty.",
        )
        d["_evidence_status"] = evidence_status
        d["_evidence_notes"] = evidence_notes
        return d


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

class OutputFormatter:
    WIDTH = 76

    # Variable labels for the summary table (primary + secondary only)
    _VAR_LABELS = {
        "fall_within_one_year":         "Q1: Falls in past year (count)",
        "feel_unsteady":                "Q2: Feels unsteady",
        "worry_fall":                   "Q3: Worries about falling",
        "pass_3kq":                     "→ Pass 3 Key Questions",
        "pass_apss":                    "Pass APSS screen",
        "have_heart_condition":         "APSS: Heart condition / stroke",
        "ex_chest_discomfort":          "APSS: Chest discomfort on exertion",
        "ex_dizzy":                     "APSS: Dizziness on exertion",
        "ex_asthma":                    "APSS: Asthma attack (past 12 mo)",
        "has_diabetes":                 "APSS: Diabetes with BSL issues",
        "unable_get_up":                "FSA: Unable to get up",
        "loc_syncope":                  "FSA: Loss of consciousness / syncope",
        "fall_with_injury":             "FSA: Fall with injury",
        "is_frail":                     "FSA: Frail (CFS ≥4)",
        "pass_fall_severity":           "→ Pass Fall Severity Assessment",
        "unsteady_gait":                "Unsteady gait",
        "fluid_restriction":            "Fluid restriction",
        "heart_failure":                "Heart failure",
    }

    def format_text(self, output: DecisionOutput) -> str:
        sep = "=" * self.WIDTH
        lines = [
            sep,
            " FALL RISK CLINICAL DECISION SUPPORT — DISCHARGE SUMMARY ANALYSIS",
            sep,
            f"\n  RISK LEVEL : {output.risk_level}",
            f"  CONDITION  : {output.condition}",
        ]

        lines += ["", "  ── Extracted Variables (primary / secondary) " + "─" * 29]
        for key, label in self._VAR_LABELS.items():
            val = output.extracted_variables.get(key)
            if val is None:
                display = "[?] Unknown"
            elif isinstance(val, bool):
                display = "Yes" if val else "No"
            else:
                display = str(val)
            lines.append(f"  {'  ' + label:<42} {display}")

        if output.warnings:
            lines += ["", "  ── Warnings / Clinician Notes " + "─" * 44]
            for w in output.warnings:
                lines += self._wrap("  [!] " + w)

        lines += [
            "",
            f"  ── Recommendations ({len(output.recommendations)}) " + "─" * 52,
        ]
        for i, rec in enumerate(output.recommendations, 1):
            lines += [
                "",
                f"  [{i}] {rec.category}",
                f"  Source: {rec.source}" + (f"  |  Grade: {rec.grade}" if rec.grade else ""),
            ]
            if rec.is_conditional:
                lines += self._wrap("  [Conditional] " + rec.condition_note)
            lines += self._wrap("  " + rec.text)

        lines += ["", sep]
        return "\n".join(lines)

    def format_json(self, output: DecisionOutput) -> str:
        extracted_variables = {
            k: v for k, v in output.extracted_variables.items()
            if not k.startswith("_")
        }
        extraction_status = output.extracted_variables.get("_evidence_status")
        extraction_evidence = output.extracted_variables.get("_evidence_notes", {})
        return json.dumps({
            "risk_level": output.risk_level,
            "condition": output.condition,
            "extracted_variables": extracted_variables,
            "extraction_status": extraction_status or self._build_extraction_status(output),
            "extraction_evidence": extraction_evidence,
            "warnings": output.warnings,
            "recommendations": [
                {
                    "index": i + 1,
                    "category": r.category,
                    "text": r.text,
                    "source": r.source,
                    "grade": r.grade,
                    "is_conditional": r.is_conditional,
                    "condition_note": r.condition_note,
                }
                for i, r in enumerate(output.recommendations)
            ],
        }, indent=2, ensure_ascii=False)

    def _build_extraction_status(self, output: DecisionOutput) -> dict:
        statuses = {}
        for key, value in output.extracted_variables.items():
            if key.startswith("_"):
                continue
            if key == "detected_FRIDs":
                statuses[key] = "extracted_positive" if value else "not_detected"
            elif value is None:
                statuses[key] = "not_documented_or_not_extractable"
            elif isinstance(value, bool):
                statuses[key] = "extracted_positive" if value else "extracted_negative"
            elif isinstance(value, int):
                statuses[key] = "extracted_positive" if value > 0 else "extracted_negative"
            else:
                statuses[key] = "extracted_value"
        return statuses

    def _wrap(self, text: str, indent: str = "     ") -> list:
        lines = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            current = ""
            for word in words:
                if not current:
                    current = word
                elif len(current) + 1 + len(word) <= self.WIDTH:
                    current += " " + word
                else:
                    lines.append(current)
                    current = indent + word
            if current:
                lines.append(current)
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class FallRiskDecisionSystem:
    """
    Main entry point.

    From free text:
        system = FallRiskDecisionSystem()
        system.analyse(discharge_summary_text)       # pretty-print to stdout
        result = system.analyse_json(text)           # returns JSON string
        output = system.analyse_raw(text)            # returns DecisionOutput

    From pre-filled variables (e.g. after a structured clinical form):
        v = PatientVariables(age=78, fall_within_one_year=2, ...)
        output = system.evaluate(v)
    """

    def __init__(self, frids_path: Optional[str] = None):
        frids_lib = _build_best_frids_library(frids_path)
        self.extractor = VariableExtractor(frids_lib)
        self.engine = RuleEngine()
        self.formatter = OutputFormatter()

    def analyse(self, discharge_summary: str) -> DecisionOutput:
        output = self._run(discharge_summary)
        print(self.formatter.format_text(output))
        return output

    def analyse_json(self, discharge_summary: str) -> str:
        return self.formatter.format_json(self._run(discharge_summary))

    def analyse_raw(self, discharge_summary: str) -> DecisionOutput:
        return self._run(discharge_summary)

    def evaluate(self, variables: PatientVariables) -> DecisionOutput:
        """Evaluate a manually-constructed PatientVariables without text extraction."""
        return self.engine.evaluate(variables)

    def _run(self, text: str) -> DecisionOutput:
        variables = self.extractor.extract(text)
        return self.engine.evaluate(variables)
