import pandas as pd
import json
from pathlib import Path
from dataclasses import fields
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from final_decision_system import PatientVariables, RuleEngine, Recommendation

imput_path = "/Users/szepan/Documents/Master's Study/BIDH5001 & BIDH5002/Code and results/mimic_IV_fall_cases_24_output_with_results.csv"

input_file = Path(imput_path)
output_dir = input_file.parent

output_csv_path = output_dir / f"{input_file.stem}_evaluation_variables.csv"

df = pd.read_csv(input_file, encoding="latin-1")
variables_col = "extraction_status_json"

def parse_json_column(value):
        if pd.isna(value):
            return {}
        
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except json.JSONDecodeError:
            return {}
        
NULL_VALUE = "null"

TRUE_VALUES = {"true", "t", "yes", "y", "1", "1.0"}
FALSE_VALUES = {"false", "flase", "f", "no", "n", "0", "0.0"}
NULL_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "not_detected",
    "not_documented",
    "not_extractable",
    "not_documented_or_not_extractable",
}


def normalise_evaluation_value(value, variable_name=None):
     if pd.isna(value):
        return NULL_VALUE

     if variable_name == "fall_within_one_year":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value) if float(value).is_integer() else value

        text = str(value).strip().lower()
        if text in NULL_VALUES:
            return NULL_VALUE
        try:
            number = float(text)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value

     if isinstance(value, bool):
        return value

     if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return value

     text = str(value).strip().lower()

     positive_values = {
          "explicit_positive",
          "inferred_positive",
          "derived_positive",
          "extracted_positive"
     }

     negative_values = {
          "explicit_negative",
          "inferred_negative",
          "derived_negative",
          "extracted_negative"
     }

     if text in positive_values:
        return True
     
     if text in negative_values:
        return False
     
     if text in TRUE_VALUES:
        return True
     
     if text in FALSE_VALUES:
        return False
     
     if text in NULL_VALUES:
        return NULL_VALUE
     
     return value


def normalise_dataframe_values(dataframe, exclude_cols):
    dataframe = dataframe.copy()
    for col in dataframe.columns:
        if col in exclude_cols:
            continue
        dataframe[col] = dataframe[col].map(
            lambda value, column=col: normalise_evaluation_value(value, column)
        )
    return dataframe


def invert_nullable_boolean(value):
    if isinstance(value, bool):
        return not value
    return NULL_VALUE


def add_take_frids_alias(dataframe):
    dataframe = dataframe.copy()
    if "take_FRIDs" not in dataframe.columns and "pass_FRIDs" in dataframe.columns:
        dataframe["take_FRIDs"] = dataframe["pass_FRIDs"].map(invert_nullable_boolean)
    if "pass_FRIDs" in dataframe.columns:
        dataframe = dataframe.drop(columns=["pass_FRIDs"])
    return dataframe


def normalise_risk_group(value):
    if pd.isna(value):
        return NULL_VALUE
    text = str(value).strip()
    if not text:
        return NULL_VALUE
    return text.split("(", 1)[0].strip()


def normalise_recommendation_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())
     
     
# Parse JSON column into dictionaries
status_dicts = df[variables_col].apply(parse_json_column)

# Expand JSON into separate columns
status_expanded = pd.json_normalize(status_dicts)

# Convert status labels into True / False / "null", preserving fall count integers.
variable_values = status_expanded.apply(
    lambda col: col.map(lambda value: normalise_evaluation_value(value, col.name))
)
variable_values = add_take_frids_alias(variable_values)

if "risk_json" in df.columns:
    fall_values = df["risk_json"].apply(
        lambda value: parse_json_column(value)
        .get("extracted_variables", {})
        .get("fall_within_one_year")
    )
    variable_values["fall_within_one_year"] = fall_values.map(
        lambda value: normalise_evaluation_value(value, "fall_within_one_year")
    )

keep_cols = [col for col in ["note_id", "hadm_id", "risk_level"] if col in df.columns]

nlp_df = pd.concat(
    [
        df[keep_cols].reset_index(drop=True),
        variable_values.reset_index(drop=True)
    ],
    axis=1
)

# Export
nlp_df.to_csv(output_csv_path, index=False)

print(f"NLP DataFrame saved to: {output_csv_path}")

# Evaluation of variables, risk level and recommendation logic (accuracy, precision, recall)
manual_path = Path ("/Users/szepan/Documents/Master's Study/BIDH5001 & BIDH5002/Code and results/Manual Evaluation Results.xlsx")
manual_df = pd.read_excel(manual_path, sheet_name="Manual Variable input", dtype={"note_id": str, "hadm_id": str})

id_cols = ["note_id", "hadm_id"]
risk_col = "risk_level"

def variable_evaluation(manual_df, nlp_df, id_cols):
    manual_df = manual_df.copy()
    nlp_df = nlp_df.copy()

    manual_df.columns = manual_df.columns.str.strip()
    nlp_df.columns = nlp_df.columns.str.strip()

    exclude_cols = set(id_cols + [risk_col])
    manual_df = normalise_dataframe_values(manual_df, exclude_cols)
    nlp_df = normalise_dataframe_values(nlp_df, exclude_cols)
    manual_df = add_take_frids_alias(manual_df)
    nlp_df = add_take_frids_alias(nlp_df)

    for col in id_cols:
        manual_df[col] = manual_df[col].astype(str).str.strip()
        nlp_df[col] = nlp_df[col].astype(str).str.strip()
    
    common_vars = (
        (set(manual_df.columns) - exclude_cols)
        .intersection(set(nlp_df.columns) - exclude_cols)
    )

    merged = manual_df.merge(
        nlp_df,
        on=id_cols,
        how="inner",
        suffixes=("_manual", "_nlp")
    )

    row_comparison_rows = []
    case_eval_rows = []
    variable_eval_rows = []

    for case_idx, row in merged.iterrows():
        note_id = row["note_id"]
        hadm_id = row["hadm_id"]

        manual_row = {
            "case_row": f"case_{case_idx + 1}_manual",
            "note_id": note_id,
            "hadm_id": hadm_id
        }

        nlp_row = {
            "case_row": f"case_{case_idx + 1}_nlp",
            "note_id": note_id,
            "hadm_id": hadm_id
        }

        case_matches = []
        bool_true = []
        bool_pred = []

        for var in common_vars:
            manual_value = row[f"{var}_manual"]
            nlp_value = row[f"{var}_nlp"]

            manual_row[var] = manual_value
            nlp_row[var] = nlp_value

            case_matches.append(manual_value == nlp_value)

            # Exclude pass_ variables and fall count from precision/recall
            if (
                not var.startswith("pass_")
                and var != "fall_within_one_year"
                and isinstance(manual_value, bool)
                and isinstance(nlp_value, bool)
            ):
                bool_true.append(manual_value)
                bool_pred.append(nlp_value)

        row_comparison_rows.append(manual_row)
        row_comparison_rows.append(nlp_row)

        risk_manual = normalise_risk_group(row.get(f"{risk_col}_manual"))
        risk_nlp = normalise_risk_group(row.get(f"{risk_col}_nlp"))

        case_eval_rows.append({
            "note_id": note_id,
            "hadm_id": hadm_id,
            "risk_group_manual": risk_manual,
            "risk_group_nlp": risk_nlp,
            "risk_group_match": risk_manual == risk_nlp,
            "case_variable_accuracy": sum(case_matches) / len(case_matches) if case_matches else None,
            "case_precision": precision_score(bool_true, bool_pred, zero_division=0) if bool_true else None,
            "case_recall": recall_score(bool_true, bool_pred, zero_division=0) if bool_true else None,
            "n_variables_compared": len(case_matches)
        })

    row_comparison_df = pd.DataFrame(row_comparison_rows)
    case_eval_df = pd.DataFrame(case_eval_rows)

    # Variable-level evaluation
    for var in common_vars:
        matches = []
        bool_true = []
        bool_pred = []

        for _, row in merged.iterrows():
            manual_value = row[f"{var}_manual"]
            nlp_value = row[f"{var}_nlp"]

            matches.append(manual_value == nlp_value)

            if (
                not var.startswith("pass_")
                and var != "fall_within_one_year"
                and isinstance(manual_value, bool)
                and isinstance(nlp_value, bool)
            ):
                bool_true.append(manual_value)
                bool_pred.append(nlp_value)

        variable_eval_rows.append({
            "variable": var,
            "variable_accuracy": sum(matches) / len(matches) if matches else None,
            "variable_precision": precision_score(bool_true, bool_pred, zero_division=0) if bool_true else None,
            "variable_recall": recall_score(bool_true, bool_pred, zero_division=0) if bool_true else None,
            "n_cases_compared": len(matches),
            "n_boolean_cases_for_precision_recall": len(bool_true)
        })

    variable_eval_df = pd.DataFrame(variable_eval_rows)

    return row_comparison_df, case_eval_df, variable_eval_df

row_comparison_df, case_eval_df, variable_eval_df = variable_evaluation(
    manual_df=manual_df,
    nlp_df=nlp_df,
    id_cols=id_cols
)

output_dir = output_csv_path.parent

row_comparison_df.to_csv(output_dir / "row_by_row_manual_vs_nlp_comparison.csv", index=False)
case_eval_df.to_csv(output_dir / "case_level_comparison_evaluation.csv", index=False)
variable_eval_df.to_csv(output_dir / "variable_level_evaluation.csv", index=False)


# Recommendation tolerance test
recommendations_path = output_dir / f"{input_file.stem.replace('_output_with_results', '')}_output_recommendations_flat.csv"
manual_recommendations_path = output_dir / "manual_input_expected_recommendations_flat.csv"
recommendation_case_eval_path = output_dir / "recommendation_case_level_evaluation.csv"
recommendation_summary_path = output_dir / "recommendation_summary_evaluation.csv"


def manual_row_to_patient_variables(row):
    field_names = {field.name for field in fields(PatientVariables)}
    alias_map = {
        "functional_diabled": "functional_disabled",
        "poor_gait_balance_muslce_strength": "poor_gait_balance_muscle_strength",
        "is_cognitive_impaired ": "is_cognitive_impaired",
    }

    values = {}
    for source_col, raw_value in row.items():
        clean_col = str(source_col).strip()
        target_col = alias_map.get(clean_col, clean_col)
        if target_col in {"note_id", "hadm_id", "risk_level"}:
            continue
        if target_col == "take_FRIDs" and "pass_FRIDs" in field_names:
            normalised = normalise_evaluation_value(raw_value, target_col)
            values["pass_FRIDs"] = invert_nullable_boolean(normalised)
            continue
        if target_col not in field_names:
            continue

        normalised = normalise_evaluation_value(raw_value, target_col)
        if normalised == NULL_VALUE:
            values[target_col] = None
        else:
            values[target_col] = normalised

    return PatientVariables(**values)


def recommendations_to_rows(note_id, hadm_id, risk_level, condition, recommendations):
    rows = []
    for index, rec in enumerate(recommendations, start=1):
        rows.append({
            "note_id": note_id,
            "hadm_id": hadm_id,
            "risk_level": risk_level,
            "condition": condition,
            "recommendation_index": index,
            "recommendation_category": rec.category,
            "recommendation_text": rec.text,
            "recommendation_source": rec.source,
            "recommendation_grade": rec.grade,
            "recommendation_is_conditional": rec.is_conditional,
            "recommendation_condition_note": rec.condition_note,
        })
    return rows


def recommendations_from_manual_risk(engine, variables, risk_group):
    warnings = engine._check_missing_core(variables)
    general = engine._general_recommendations(variables, warnings)

    if risk_group == "Low Risk":
        recs, extra_warnings = engine._low_risk_exercise(variables)
        return general + recs, "Manual risk group: Low Risk", warnings + extra_warnings

    if risk_group == "Intermediate Risk":
        recs, extra_warnings = engine._intermediate_risk_rules(variables)
        return general + recs, "Manual risk group: Intermediate Risk", warnings + extra_warnings

    if risk_group == "High Risk":
        recs, extra_warnings = engine._high_risk_rules(variables)
        return general + recs, "Manual risk group: High Risk", warnings + extra_warnings

    if risk_group == "Intermediate / High Risk":
        assessment = Recommendation(
            category="Assessment Required",
            text=(
                "Complete Fall Severity Assessment (unable to get up, number of falls in a year, "
                "loss of consciousness, fall with injury, frailty status) to determine the "
                "appropriate risk pathway."
            ),
            source="System",
        )
        return general + [assessment], "Manual risk group: Intermediate / High Risk", warnings

    assessment = Recommendation(
        category="Assessment Required",
        text=(
            "Unable to determine fall risk level automatically. Please manually assess the "
            "3 Key Questions: Q1 falls in the past year; Q2 feels unsteady; Q3 worries about falling."
        ),
        source="System",
    )
    return general + [assessment], "Manual risk group: Indeterminate", warnings


def build_manual_expected_recommendations(manual_df):
    engine = RuleEngine()
    manual_df = manual_df.copy()
    manual_df.columns = manual_df.columns.str.strip()
    exclude_cols = set(id_cols + [risk_col])
    manual_df = normalise_dataframe_values(manual_df, exclude_cols)
    manual_df = add_take_frids_alias(manual_df)

    rows = []
    for _, row in manual_df.iterrows():
        note_id = str(row.get("note_id")).strip()
        hadm_id = str(row.get("hadm_id")).strip()
        risk_group = normalise_risk_group(row.get(risk_col))
        variables = manual_row_to_patient_variables(row)
        recommendations, condition, _warnings = recommendations_from_manual_risk(
            engine, variables, risk_group
        )
        rows.extend(recommendations_to_rows(
            note_id, hadm_id, risk_group, condition, recommendations
        ))

    return pd.DataFrame(rows)


def recommendation_key(row):
    return (
        normalise_recommendation_text(row.get("recommendation_category")),
        normalise_recommendation_text(row.get("recommendation_text")),
    )


def recommendation_tolerance_evaluation(expected_df, actual_df):
    expected_df = expected_df.copy()
    actual_df = actual_df.copy()

    for frame in [expected_df, actual_df]:
        for col in id_cols:
            frame[col] = frame[col].astype(str).str.strip()

    case_ids = (
        pd.concat([expected_df[id_cols], actual_df[id_cols]], ignore_index=True)
        .drop_duplicates()
        .sort_values(id_cols)
    )

    case_rows = []
    total_expected = 0
    total_actual = 0
    total_matched = 0

    for _, case in case_ids.iterrows():
        note_id = case["note_id"]
        hadm_id = case["hadm_id"]

        expected_case = expected_df[
            (expected_df["note_id"] == note_id) & (expected_df["hadm_id"] == hadm_id)
        ]
        actual_case = actual_df[
            (actual_df["note_id"] == note_id) & (actual_df["hadm_id"] == hadm_id)
        ]

        expected_set = {recommendation_key(row) for _, row in expected_case.iterrows()}
        actual_set = {recommendation_key(row) for _, row in actual_case.iterrows()}
        matched = expected_set & actual_set
        union = expected_set | actual_set

        expected_count = len(expected_set)
        actual_count = len(actual_set)
        matched_count = len(matched)
        precision = matched_count / actual_count if actual_count else None
        recall = matched_count / expected_count if expected_count else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        jaccard = matched_count / len(union) if union else 1.0

        total_expected += expected_count
        total_actual += actual_count
        total_matched += matched_count

        missing = expected_set - actual_set
        extra = actual_set - expected_set
        case_rows.append({
            "note_id": note_id,
            "hadm_id": hadm_id,
            "expected_recommendation_count": expected_count,
            "actual_recommendation_count": actual_count,
            "matched_recommendation_count": matched_count,
            "recommendation_precision": precision,
            "recommendation_recall": recall,
            "recommendation_f1": f1,
            "recommendation_accuracy_jaccard": jaccard,
            "recommendation_exact_match": expected_set == actual_set,
            "missing_recommendations": " || ".join(
                f"{category}: {text}" for category, text in sorted(missing)
            ),
            "extra_recommendations": " || ".join(
                f"{category}: {text}" for category, text in sorted(extra)
            ),
        })

    overall_precision = total_matched / total_actual if total_actual else None
    overall_recall = total_matched / total_expected if total_expected else None
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if overall_precision is not None
        and overall_recall is not None
        and (overall_precision + overall_recall) > 0
        else None
    )

    summary_df = pd.DataFrame([{
        "n_cases": len(case_rows),
        "total_expected_recommendations": total_expected,
        "total_actual_recommendations": total_actual,
        "total_matched_recommendations": total_matched,
        "overall_recommendation_precision": overall_precision,
        "overall_recommendation_recall": overall_recall,
        "overall_recommendation_f1": overall_f1,
        "mean_case_recommendation_accuracy_jaccard": (
            pd.DataFrame(case_rows)["recommendation_accuracy_jaccard"].mean()
            if case_rows else None
        ),
        "exact_match_case_rate": (
            pd.DataFrame(case_rows)["recommendation_exact_match"].mean()
            if case_rows else None
        ),
    }])

    return pd.DataFrame(case_rows), summary_df


if recommendations_path.exists():
    manual_expected_recommendations_df = build_manual_expected_recommendations(manual_df)
    actual_recommendations_df = pd.read_csv(
        recommendations_path, encoding="utf-8-sig", dtype={"note_id": str, "hadm_id": str}
    )
    recommendation_case_eval_df, recommendation_summary_df = recommendation_tolerance_evaluation(
        manual_expected_recommendations_df, actual_recommendations_df
    )

    manual_expected_recommendations_df.to_csv(manual_recommendations_path, index=False)
    recommendation_case_eval_df.to_csv(recommendation_case_eval_path, index=False)
    recommendation_summary_df.to_csv(recommendation_summary_path, index=False)

    print(f"Manual-input expected recommendations saved to: {manual_recommendations_path}")
    print(f"Recommendation case-level evaluation saved to: {recommendation_case_eval_path}")
    print(f"Recommendation summary evaluation saved to: {recommendation_summary_path}")
else:
    print(f"Recommendation flat file not found, skipped recommendation evaluation: {recommendations_path}")
