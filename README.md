# Fall Risk Clinical Decision Support System

Rule-based natural language processing and clinical decision support for
identifying fall-risk factors in hospital discharge summaries, stratifying
patients by fall risk, and generating personalised fall-prevention
recommendations.

This repository contains the source code for a BIDH5001/BIDH5002 capstone
project. The system was developed for research and educational evaluation using
authorised MIMIC-IV discharge summaries.

> **Important:** This is a research prototype, not a validated medical device.
> Its output must not replace clinical judgement, diagnosis, or treatment.

## Project Objective

The project addresses the transition from emergency or inpatient care to
community fall-prevention services. It aims to:

1. Extract structured fall-risk variables from free-text discharge summaries.
2. Preserve uncertainty when information is absent or cannot be extracted.
3. Assign a low, intermediate, high, or indeterminate fall-risk pathway.
4. Produce personalised recommendations based on documented risk factors.
5. Compare NLP outputs against manual review at variable, risk-group, and
   recommendation levels.

Rule-based NLP was selected because it is transparent, auditable, and suitable
for a small manually reviewed dataset.

## System Architecture

```text
Discharge summary
       |
       v
Section-aware regex and lexicon extraction
       |
       v
PatientVariables
  - explicit values
  - inferred values
  - derived values
  - undocumented values
       |
       v
Rule-based risk stratification
       |
       v
Personalised recommendation engine
       |
       v
JSON, CSV, and evaluation outputs
```

### 1. Variable extraction

`VariableExtractor` maps discharge-summary text to `PatientVariables`. It uses:

- regular expressions and clinical keyword lists;
- negation handling;
- section-aware extraction;
- discharge-medication matching for fall-risk-increasing drugs (FRIDs);
- conservative `None` values when evidence is unavailable;
- an inference layer for selected mobility and functional variables.

The variables cover:

- the Three Key Questions: fall history, unsteadiness, and worry about falling;
- fall severity: repeated falls, inability to get up, loss of
  consciousness/syncope, fall injury, and frailty;
- gait, balance, mobility aids, and functional limitation;
- exercise safety screening;
- cognition, cardiovascular conditions, dizziness, vision, hearing, foot
  problems, pain, continence, nutrition, and home-safety needs;
- discharge medications that may increase fall risk.

### 2. Evidence status

Each variable is accompanied by an evidence status where possible:

- `explicit_positive` / `explicit_negative`
- `inferred_positive` / `inferred_negative`
- `derived_positive` / `derived_negative`
- `not_documented_or_not_extractable`

The intended precedence is:

```text
explicit evidence > inferred evidence > derived evidence > undocumented
```

This makes missing information visible rather than silently converting it to a
negative result.

### 3. Risk stratification

The rule engine applies the following main pathways:

| Three Key Questions | Fall severity | Gait/mobility evidence | Result |
|---|---|---|---|
| Negative | Any | Any | Low Risk |
| Positive | Non-severe | Steady gait | Low Risk |
| Positive | Non-severe | Unsteady/poor gait | Intermediate Risk |
| Positive | Severe | Any | High Risk |
| Positive | Unknown | Any | Intermediate / High Risk pending assessment |
| Unknown | Severe | Any | High Risk inferred from documented severity |
| Unknown | Not severe/unknown | Mobility concern present | Intermediate Risk inferred from mobility concerns |
| Unknown | Unknown | No mobility concern | Indeterminate |

Fall count is an ordinal text-derived proxy rather than an exact epidemiological
count:

- `0`: explicit evidence of no fall history;
- `1`: an index fall associated with the admission;
- `2`: recent, repeated, recurrent, or multiple falls;
- `None`: no fall-related evidence was extractable.

### 4. Recommendation engine

Recommendations are selected from rule-based fall-prevention domains including:

- exercise and strength/balance training;
- medication and FRIDs review;
- environmental and home-safety modification;
- vision, hearing, foot, and footwear assessment;
- cardiovascular, dizziness, and syncope follow-up;
- cognition, continence, nutrition, vitamin D, calcium, and fracture risk;
- functional assessment, walking aids, and community support;
- fall emergency planning.

Recommendation records include category, recommendation text, source, evidence
grade where available, and conditional-use notes.

## Repository Files

| File | Purpose |
|---|---|
| `final_decision_system.py` | Main variable extractor, inference layer, risk engine, recommendation engine, and output formatter |
| `final_discharge_demo.py` | Batch processing of discharge summaries and export of risk, evidence, and recommendation outputs |
| `evaluation.py` | Manual-versus-NLP variable evaluation, risk-group comparison, and recommendation tolerance testing |
| `frids_library.json` | Structured FRIDs medication catalogue grouped by medication class |
| `frids_loader_snippet.py` | Minimal example for loading and matching the FRIDs catalogue |

## Requirements

- Python 3.10 or later
- `pandas`
- `openpyxl`
- `scikit-learn`

Install the dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Basic Usage

Analyse one discharge summary:

```python
from final_decision_system import FallRiskDecisionSystem

system = FallRiskDecisionSystem()

text = """
Discharge summary text supplied by an authorised user.
"""

result = system.analyse_raw(text)

print(result.risk_level)
print(result.condition)

for recommendation in result.recommendations:
    print(recommendation.category, recommendation.text)
```

Obtain a JSON representation:

```python
json_result = system.analyse_json(text)
print(json_result)
```

Evaluate manually populated variables without NLP:

```python
from final_decision_system import FallRiskDecisionSystem, PatientVariables

variables = PatientVariables(
    fall_within_one_year=1,
    feel_unsteady=True,
    fall_with_injury=False,
)

output = FallRiskDecisionSystem().evaluate(variables)
print(output.risk_level)
```

## Batch Processing

`final_discharge_demo.py` expects a local CSV containing a `text` column and,
where available, `note_id` and `hadm_id`.

Before running it, update `input_path` and the manual-template path for the local
environment. Then run:

```bash
python3 final_discharge_demo.py
```

The script can generate:

- a detailed per-case output CSV;
- risk-level counts;
- flattened recommendation rows;
- a manual-evaluation workbook populated with NLP extraction values.

The current scripts use explicit local paths for reproducibility in the original
research environment. A future improvement would replace these with command-line
arguments or a configuration file.

## Evaluation

`evaluation.py` compares NLP output with an independently completed manual
evaluation workbook.

### Variable evaluation

The evaluation:

- normalises spreadsheet Boolean values while preserving `True`, `False`, and
  `null`;
- preserves the integer/ordinal value of `fall_within_one_year`;
- compares variables per case and across cases;
- reports accuracy, precision, and recall;
- excludes derived `pass_*` variables and the ordinal fall count from Boolean
  precision/recall calculations;
- evaluates `take_FRIDs`, where `True` means the patient is taking at least one
  matched FRID.

### Risk-group evaluation

Risk labels are normalised to the main group before comparison. For example:

```text
Intermediate Risk (inferred from mobility concerns)
```

is compared as:

```text
Intermediate Risk
```

### Recommendation tolerance test

The recommendation test:

1. converts each manual variable row into `PatientVariables`;
2. runs the recommendation rules using the manual risk group;
3. compares expected recommendations with recommendations generated from NLP
   extraction;
4. reports per-case precision, recall, F1, Jaccard similarity, and exact match;
5. reports aggregate recommendation-level performance.

Run the evaluation after updating the local input paths:

```bash
python3 evaluation.py
```

## Data and Privacy

MIMIC-IV data is credentialed-access data governed by its data-use agreement.
No MIMIC-IV discharge summaries, patient-level input files, manual annotations,
or generated patient-level outputs are included in this repository.

Authorised users should obtain MIMIC-IV independently, place the required files
in their local environment, and update the configured paths. Do not commit or
publish any restricted clinical text or derived patient-level records.

## Known Limitations

- Regex rules are sensitive to wording, abbreviations, spelling, and section
  structure.
- Whole-document matching can confuse historical, hypothetical, negated, and
  discharge-instruction contexts.
- Evidence expressed across multiple sentences may not be linked correctly.
- Missing documentation and extraction failure both commonly result in `None`.
- The ordinal fall-count proxy does not represent an exact number of falls.
- Inference rules improve coverage but can introduce false positives.
- Evaluation was performed on a small manually reviewed sample and should not
  be interpreted as evidence of clinical validation or generalisability.
- Recommendations require clinician review for contraindications, patient
  preferences, feasibility, and local referral pathways.

## Reproducibility Notes

For a transparent review:

1. Inspect `PatientVariables` to see the complete variable schema.
2. Inspect `VariableExtractor` for extraction and inference rules.
3. Inspect `RuleEngine.evaluate()` for risk-pathway logic.
4. Inspect the recommendation methods in `RuleEngine`.
5. Run `final_discharge_demo.py` using authorised local data.
6. Run `evaluation.py` with the corresponding manual evaluation workbook.
7. Review row-level comparison files before interpreting aggregate metrics.

## Academic and Clinical Disclaimer

This project is provided for academic assessment and research demonstration.
It has not undergone prospective clinical validation, regulatory review, or
deployment testing. Any clinical implementation would require external
validation, human-factors testing, privacy and security review, governance,
ongoing monitoring, and approval by the relevant health service.

