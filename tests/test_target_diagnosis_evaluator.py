from __future__ import annotations

import evaluate_target_ecgfeat_diagnosis as evaluator


def test_ptbxl_manifest_indexes_500_hz_filename_hr(tmp_path) -> None:
    dataset_dir = tmp_path / "records"
    metadata_dir = tmp_path / "metadata"
    dataset_dir.mkdir()
    metadata_dir.mkdir()
    (dataset_dir / "01000_hr.hea").write_text(
        "01000_hr 12 500 5000\n",
        encoding="utf-8",
    )
    (metadata_dir / "scp_statements.csv").write_text(
        ",description,diagnostic\n"
        "NORM,normal ECG,1.0\n",
        encoding="utf-8",
    )
    (metadata_dir / "ptbxl_database.csv").write_text(
        "ecg_id,age,sex,scp_codes,filename_lr,filename_hr\n"
        '1000,53,0,"{\'NORM\': 100.0}",'
        "records100/01000/01000_lr,records500/01000/01000_hr\n",
        encoding="utf-8",
    )

    manifest, _ = evaluator._load_ptbxl_manifest(
        dataset_dir,
        metadata_dir,
    )

    assert len(manifest) == 1
    assert manifest[0]["record"] == "01000_hr"
    assert manifest[0]["reference_codes_active"] == ["NORM"]
    assert manifest[0]["record_path"].endswith("01000_hr")


def test_local_unspecified_av_block_is_not_repeated_as_specific_av_block() -> None:
    categories = evaluator._reference_categories(
        evaluator.DATASET_LOCAL, {"233917008"}
    )
    unsupported, details = evaluator._unsupported_reference_details(
        evaluator.DATASET_LOCAL,
        {"233917008"},
        {"233917008": "Atrioventricular block"},
    )

    assert categories == set()
    assert unsupported == ["233917008"]
    assert "不能与一度、二度和完全性房室阻滞重复计分" in details[0]


def test_ambiguous_local_left_conduction_code_accepts_fascicular_pattern() -> None:
    references = evaluator._reference_categories(
        evaluator.DATASET_LOCAL, {"164909002"}
    )
    predictions = evaluator._prediction_categories(
        evaluator.DATASET_LOCAL, {"lafb_pattern"}
    )

    assert references == {"ambiguous_left_conduction_family"}
    assert predictions == {"ambiguous_left_conduction_family"}


def test_generic_rate_observations_are_not_direct_sinus_diagnoses() -> None:
    lookup = evaluator._category_lookup()

    assert lookup["sinus_bradycardia"].semantic_scope == "direct"
    assert lookup["sinus_tachycardia"].semantic_scope == "direct"
    assert lookup["sinus_bradycardia"].prediction_codes == frozenset(
        {"sinus_bradycardia"}
    )
    assert lookup["sinus_tachycardia"].prediction_codes == frozenset(
        {"sinus_tachycardia"}
    )


def test_st_and_marked_qt_outputs_are_scored_in_their_families() -> None:
    predictions = evaluator._prediction_categories(
        evaluator.DATASET_PTBXL,
        {
            "st_depression",
            "markedly_prolonged_qt",
        },
    )

    assert "ischemia_or_st_abnormality" in predictions
    assert "prolonged_qt" in predictions


def test_probable_af_is_scored_but_possible_flutter_is_observation_only() -> None:
    af = evaluator._prediction_categories(
        evaluator.DATASET_LOCAL,
        {"probable_atrial_fibrillation_pattern"},
    )
    flutter = evaluator._prediction_categories(
        evaluator.DATASET_LOCAL,
        {"possible_atrial_flutter_pattern"},
    )

    assert af == {"atrial_fibrillation"}
    assert flutter == set()
    assert (
        "possible_atrial_flutter_pattern"
        in evaluator.NONCOMPARABLE_OBSERVATION_CODES
    )


def test_pac_or_pvc_is_not_mapped_to_bigeminy_or_trigeminy() -> None:
    ordinary_ectopy = evaluator._prediction_categories(
        evaluator.DATASET_PTBXL,
        {"premature_atrial_complexes", "premature_ventricular_complexes"},
    )
    patterned_ectopy = evaluator._prediction_categories(
        evaluator.DATASET_PTBXL,
        {"ventricular_bigeminy_pattern"},
    )

    assert "unspecified_ectopy_pattern" not in ordinary_ectopy
    assert "unspecified_ectopy_pattern" in patterned_ectopy


def test_dataset_summary_reports_direct_and_all_mapped_metrics_separately() -> None:
    metrics = [
        {
            "dataset": evaluator.DATASET_LOCAL,
            "semantic_scope": "direct",
            "tp": 2,
            "fp": 1,
            "fn": 1,
        },
        {
            "dataset": evaluator.DATASET_LOCAL,
            "semantic_scope": "broad",
            "tp": 3,
            "fp": 4,
            "fn": 2,
        },
    ]
    raw = [
        {
            "dataset": evaluator.DATASET_LOCAL,
            "extraction_status": "ok",
            "runtime_seconds": 1.0,
        }
    ]

    summary = evaluator._dataset_summary(
        raw,
        [],
        metrics,
        evaluator.DATASET_LOCAL,
    )

    assert summary["direct_micro_tp"] == 2
    assert summary["direct_micro_fp"] == 1
    assert summary["direct_micro_fn"] == 1
    assert summary["micro_tp"] == 5
    assert summary["micro_fp"] == 5
    assert summary["micro_fn"] == 3
