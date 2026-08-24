from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import batch_medgemma_diagnostics as batch_script


class BatchMedGemmaDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _pass_gate() -> dict:
        return {
            "state": "pass",
            "stop_reasons": [],
            "partial_reasons": [],
            "allowed_domains": ["all"],
            "suppressed_domains": [],
        }

    def test_layered_is_default_and_legacy_requires_explicit_acknowledgement(self) -> None:
        parser = batch_script.build_arg_parser()

        defaults = parser.parse_args([])
        legacy = parser.parse_args(["--method", "legacy"])

        self.assertEqual("layered", defaults.method)
        self.assertFalse(defaults.allow_non_independent_legacy)
        with self.assertRaisesRegex(RuntimeError, "non-independent"):
            batch_script.run_batch(legacy)

    def test_iter_samples_pairs_feature_and_report_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "all_diseases_ecgfeat" / "ami" / "generated"
            split_dir.mkdir(parents=True)
            (split_dir / "001_features.json").write_text("{}", encoding="utf-8")
            (split_dir / "001_report.txt").write_text("report", encoding="utf-8")

            samples = list(batch_script.iter_samples(root / "all_diseases_ecgfeat", ("generated", "real")))

            self.assertEqual(1, len(samples))
            self.assertEqual("ami", samples[0].disease_dir_label)
            self.assertEqual("generated", samples[0].split_name)
            self.assertEqual("001", samples[0].sample_id)

    def test_process_sample_writes_both_local_and_mirror_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "all_diseases_ecgfeat"
            mirror_root = root / "all_diseases_medgemma"
            split_dir = input_root / "ami" / "generated"
            split_dir.mkdir(parents=True)
            features_path = split_dir / "001_features.json"
            report_path = split_dir / "001_report.txt"
            features_path.write_text(
                json.dumps(
                    {
                        "global_features": {},
                        "interpretation": {},
                        "groups": {},
                        "quality": {},
                        "representative_leads": {},
                        "metadata": {
                            "patient_meta": {"age": 50},
                            "diagnostic_gate": self._pass_gate(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            report_path.write_text(
                "Record ID : ami_generated_001\nHx : disease=ami\nHeart Rate : 88 bpm",
                encoding="utf-8",
            )

            sample = batch_script.SampleJob(
                disease_dir_label="ami",
                split_name="generated",
                sample_id="001",
                features_path=features_path,
                report_path=report_path,
                local_output_dir=split_dir,
                mirror_output_dir=mirror_root / "ami" / "generated",
            )

            result = batch_script.process_sample(
                sample=sample,
                language="en",
                clinician_notes="",
                generate_text_fn=lambda prompt: """### Most Likely Diagnosis
- Acute myocardial infarction

### Diagnostic Rationale
- ST-T abnormality.

### Similar Diagnoses (ranked by similarity)
1. Diagnosis: Ischemia
   Shared features: ST change
   Key differentiator: No infarction wording
2. Diagnosis: Old myocardial infarction
   Shared features: Ischemic evidence
   Key differentiator: Chronicity
3. Diagnosis: Left ventricular hypertrophy
   Shared features: Repolarization change
   Key differentiator: Voltage pattern

### Uncertainty and Review Priorities
- Review lead quality.""",
            )

            self.assertEqual("acute myocardial infarction", result["parsed"]["top1_normalized"])
            self.assertTrue((split_dir / "001_medgemma.json").exists())
            self.assertTrue((split_dir / "001_medgemma.txt").exists())
            self.assertTrue((mirror_root / "ami" / "generated" / "001_medgemma.json").exists())
            self.assertTrue((mirror_root / "ami" / "generated" / "001_medgemma.txt").exists())

            saved = json.loads((split_dir / "001_medgemma.json").read_text(encoding="utf-8"))
            self.assertNotIn("disease=ami", saved["sanitized_context"].lower())
            self.assertNotIn("ami_generated_001", saved["prompt"].lower())

    def test_process_sample_layered_writes_layered_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "all_diseases_ecgfeat"
            mirror_root = root / "all_diseases_medgemma"
            split_dir = input_root / "norm" / "generated"
            split_dir.mkdir(parents=True)
            features_path = split_dir / "001_features.json"
            features_path.write_text(
                json.dumps(
                    {
                        "global_features": {"heart_rate_bpm": 70.0, "qrs_ms": 90.0},
                        "interpretation": {"pathological_q_leads": {}},
                        "groups": {},
                        "quality": {},
                        "representative_leads": {},
                        "metadata": {
                            "patient_meta": {"age": 50},
                            "diagnostic_gate": self._pass_gate(),
                        },
                    }
                ),
                encoding="utf-8",
            )

            sample = batch_script.SampleJob(
                disease_dir_label="norm",
                split_name="generated",
                sample_id="001",
                features_path=features_path,
                report_path=None,
                local_output_dir=split_dir,
                mirror_output_dir=mirror_root / "norm" / "generated",
            )

            layered_output = (
                "### L0. Signal Quality Gate\n- ok\n\n### L1. Rhythm\n- sinus\n\n"
                "### L2. Conduction & Intervals\n- normal\n\n### L3. Axis & Chamber Morphology\n- normal\n\n"
                "### L4. ST-T / Ischemia & Infarction\n- none\n\n### L5. Synthesis\n"
                "#### Most Likely Diagnosis\n- Normal ECG\n\n#### Diagnostic Rationale (layer references)\n- ok\n\n"
                "#### Similar Diagnoses (ranked by similarity)\n"
                "1. Diagnosis: Sinus rhythm\n   Shared features: a\n   Key differentiator: b\n"
                "2. Diagnosis: Athletic heart\n   Shared features: a\n   Key differentiator: b\n"
                "3. Diagnosis: Early repol\n   Shared features: a\n   Key differentiator: b\n\n"
                "#### Layer Conflicts\n- none\n\n#### Uncertainty and Review Priorities\n- none"
            )

            result = batch_script.process_sample_layered(
                sample=sample,
                language="en",
                clinician_notes="",
                generate_text_fn=lambda prompt: layered_output,
            )

            self.assertEqual("normal ecg", result["parsed"]["top1_normalized"])
            self.assertFalse(result["revised"])
            self.assertTrue((split_dir / "001_medgemma_layered.json").exists())
            self.assertTrue((split_dir / "001_medgemma_layered.txt").exists())
            self.assertTrue((mirror_root / "norm" / "generated" / "001_medgemma_layered.json").exists())

    def test_skip_existing_loads_valid_results_into_complete_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "all_diseases_ecgfeat"
            mirror_root = root / "all_diseases_medgemma"
            cases = (
                ("ami", "001", "Acute myocardial infarction", "acute myocardial infarction"),
                ("norm", "002", "Normal ECG", "normal ecg"),
            )
            for disease, sample_id, top1_raw, top1_normalized in cases:
                split_dir = input_root / disease / "generated"
                split_dir.mkdir(parents=True, exist_ok=True)
                (split_dir / f"{sample_id}_features.json").write_text(
                    "{}", encoding="utf-8"
                )
                existing = {
                    "sample_id": sample_id,
                    "split": "generated",
                    "ground_truth_dir_label": disease,
                    "ground_truth_normalized_label": batch_script.canonical_label_for_dir(
                        disease
                    ),
                    "features_sha256": batch_script._file_sha256(
                        split_dir / f"{sample_id}_features.json"
                    ),
                    "report_sha256": None,
                    "evaluation_method": "layered_measurement_only_gated",
                    "independent_evaluation": True,
                    "parse_status": "ok",
                    "parsed": {
                        "top1_raw": top1_raw,
                        "top1_normalized": top1_normalized,
                        "similar_raw": [],
                        "similar_normalized": [],
                    },
                }
                (split_dir / f"{sample_id}_medgemma_layered.json").write_text(
                    json.dumps(existing), encoding="utf-8"
                )

            args = batch_script.build_arg_parser().parse_args(
                [
                    "--input-dir",
                    str(input_root),
                    "--mirror-output-dir",
                    str(mirror_root),
                    "--splits",
                    "generated",
                    "--skip-existing",
                ]
            )
            execution_signature = batch_script.build_execution_signature(args)
            for disease, sample_id, _top1_raw, _top1_normalized in cases:
                result_path = (
                    input_root
                    / disease
                    / "generated"
                    / f"{sample_id}_medgemma_layered.json"
                )
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                payload["execution_signature"] = execution_signature
                result_path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                batch_script,
                "build_generate_text_fn",
                return_value=lambda _prompt: self.fail(
                    "valid skipped results must not invoke the model"
                ),
            ) as model_factory:
                status = batch_script.run_batch(args)
            model_factory.assert_not_called()

            metrics = json.loads(
                (mirror_root / "summary" / "normalized_metrics.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(0, status)
        self.assertEqual(2, metrics["combined"]["count"])
        self.assertEqual(1.0, metrics["combined"]["top1_accuracy"])
        self.assertEqual(1.0, metrics["operational"]["operational_coverage"])
        self.assertTrue(metrics["operational"]["all_samples_completed"])

    def test_batch_model_failure_is_nonzero_and_reported_as_incomplete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "all_diseases_ecgfeat"
            mirror_root = root / "all_diseases_medgemma"
            split_dir = input_root / "norm" / "generated"
            split_dir.mkdir(parents=True)
            (split_dir / "001_features.json").write_text(
                json.dumps(
                    {
                        "global_features": {"heart_rate_bpm": 70.0},
                        "interpretation": {},
                        "groups": {},
                        "quality": {},
                        "representative_leads": {},
                        "metadata": {
                            "patient_meta": {"age": 50},
                            "diagnostic_gate": self._pass_gate(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = batch_script.build_arg_parser().parse_args(
                [
                    "--input-dir",
                    str(input_root),
                    "--mirror-output-dir",
                    str(mirror_root),
                    "--splits",
                    "generated",
                ]
            )

            def fail_generate(_prompt: str) -> str:
                raise RuntimeError("model unavailable")

            with mock.patch.object(
                batch_script,
                "build_generate_text_fn",
                return_value=fail_generate,
            ):
                status = batch_script.run_batch(args)
            metrics = json.loads(
                (mirror_root / "summary" / "normalized_metrics.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(1, status)
        self.assertEqual(1, metrics["operational"]["requested_count"])
        self.assertEqual(1, metrics["operational"]["failed_count"])
        self.assertEqual(0.0, metrics["operational"]["operational_coverage"])
        self.assertFalse(metrics["operational"]["all_samples_completed"])

    def test_legacy_stop_gate_abstains_without_invoking_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "ami" / "generated"
            split_dir.mkdir(parents=True)
            features_path = split_dir / "001_features.json"
            features_path.write_text(
                json.dumps(
                    {
                        "global_features": {},
                        "interpretation": {"probable_af": True},
                        "groups": {},
                        "quality": {},
                        "representative_leads": {},
                        "metadata": {
                            "diagnostic_gate": {
                                "state": "stop",
                                "stop_reasons": ["fewer_than_3_detected_beats"],
                                "partial_reasons": [],
                                "allowed_domains": [],
                                "suppressed_domains": [],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            sample = batch_script.SampleJob(
                disease_dir_label="ami",
                split_name="generated",
                sample_id="001",
                features_path=features_path,
                report_path=None,
                local_output_dir=split_dir,
                mirror_output_dir=root / "mirror",
            )

            result = batch_script.process_sample(
                sample,
                "en",
                "",
                lambda _prompt: self.fail("model must not run through a stop gate"),
            )

        self.assertEqual("abstained_diagnostic_gate", result["parse_status"])
        self.assertIsNone(result["parsed"]["top1_raw"])
        self.assertFalse(result["independent_evaluation"])

    def test_legacy_partial_gate_abstains_without_invoking_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "ami" / "generated"
            split_dir.mkdir(parents=True)
            features_path = split_dir / "001_features.json"
            features_path.write_text(
                json.dumps(
                    {
                        "global_features": {},
                        "interpretation": {"probable_af": True},
                        "groups": {},
                        "quality": {},
                        "representative_leads": {},
                        "metadata": {
                            "diagnostic_gate": {
                                "state": "partial",
                                "stop_reasons": [],
                                "partial_reasons": ["morphology_unreliable"],
                                "allowed_domains": ["rhythm"],
                                "suppressed_domains": ["morphology"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            sample = batch_script.SampleJob(
                disease_dir_label="ami",
                split_name="generated",
                sample_id="001",
                features_path=features_path,
                report_path=None,
                local_output_dir=split_dir,
                mirror_output_dir=root / "mirror",
            )
            calls: list[str] = []

            result = batch_script.process_sample(
                sample,
                "en",
                "",
                lambda prompt: calls.append(prompt) or "must not run",
            )

        self.assertEqual([], calls)
        self.assertEqual("abstained_diagnostic_gate", result["parse_status"])
        self.assertEqual("diagnostic_gate_not_pass", result["parsed"]["abstention_reason"])
        self.assertIn("PARTIAL", result["raw_model_output"])
        self.assertFalse(result["independent_evaluation"])

    def test_write_summary_outputs_includes_raw_and_normalized_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_dir = root / "summary"
            rows = [
                {
                    "split": "generated",
                    "ground_truth_dir_label": "ami",
                    "ground_truth_normalized_label": "acute myocardial infarction",
                    "top1_raw": "Acute myocardial infarction",
                    "top1_normalized": "acute myocardial infarction",
                    "similar_raw": ["Ischemia", "Old myocardial infarction", "Left ventricular hypertrophy"],
                    "similar_normalized": ["ischemia", "old myocardial infarction", "left ventricular hypertrophy"],
                },
                {
                    "split": "real",
                    "ground_truth_dir_label": "crbbb",
                    "ground_truth_normalized_label": "complete right bundle branch block",
                    "top1_raw": "Right bundle branch block",
                    "top1_normalized": "complete right bundle branch block",
                    "similar_raw": ["Intraventricular conduction delay"],
                    "similar_normalized": [None],
                },
            ]

            batch_script.write_summary_outputs(rows, summary_dir)

            self.assertTrue((summary_dir / "per_sample.csv").exists())
            self.assertTrue((summary_dir / "raw_counts.json").exists())
            self.assertTrue((summary_dir / "normalized_metrics.json").exists())
            self.assertTrue((summary_dir / "confusion_top1.csv").exists())
            self.assertTrue((summary_dir / "confusion_top3.csv").exists())

            metrics = json.loads((summary_dir / "normalized_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(1.0, metrics["by_split"]["generated"]["top1_accuracy"])
            self.assertEqual(1.0, metrics["by_split"]["real"]["top1_accuracy"])
            self.assertEqual(1.0, metrics["by_split"]["generated"]["top3_hit_rate"])

    def test_zero_sample_run_is_nonzero_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "all_diseases_ecgfeat"
            input_root.mkdir()
            mirror_root = root / "all_diseases_medgemma"

            with mock.patch.object(batch_script, "build_generate_text_fn") as model_factory:
                status = batch_script.main(
                    [
                        "--input-dir",
                        str(input_root),
                        "--mirror-output-dir",
                        str(mirror_root),
                        "--splits",
                        "generated",
                    ]
                )

            self.assertEqual(1, status)
            model_factory.assert_not_called()
            self.assertFalse((mirror_root / "summary").exists())

    def test_reusable_result_is_bound_to_full_execution_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "inputs" / "norm" / "generated"
            split_dir.mkdir(parents=True)
            features_path = split_dir / "001_features.json"
            features_path.write_text("{}", encoding="utf-8")
            sample = batch_script.SampleJob(
                disease_dir_label="norm",
                split_name="generated",
                sample_id="001",
                features_path=features_path,
                report_path=None,
                local_output_dir=split_dir,
                mirror_output_dir=root / "mirror" / "norm" / "generated",
            )
            parser = batch_script.build_arg_parser()
            args_a = parser.parse_args(["--model-path", "model-a"])
            args_b = parser.parse_args(["--model-path", "model-b"])
            signature_a = batch_script.build_execution_signature(args_a)
            signature_b = batch_script.build_execution_signature(args_b)
            payload = {
                "sample_id": "001",
                "split": "generated",
                "ground_truth_dir_label": "norm",
                "ground_truth_normalized_label": "normal ecg",
                "features_sha256": batch_script._file_sha256(features_path),
                "report_sha256": None,
                "evaluation_method": "layered_measurement_only_gated",
                "independent_evaluation": True,
                "parse_status": "ok",
                "parsed": {
                    "top1_raw": "Normal ECG",
                    "top1_normalized": "normal ecg",
                    "similar_raw": [],
                    "similar_normalized": [],
                },
                "execution_signature": signature_a,
            }

            row = batch_script._summary_row_from_result(
                payload, sample, "layered", signature_a
            )
            self.assertEqual("normal ecg", row["top1_normalized"])
            with self.assertRaisesRegex(ValueError, "signature is stale"):
                batch_script._summary_row_from_result(
                    payload, sample, "layered", signature_b
                )

            self.assertEqual(
                batch_script.DIAGNOSTIC_GATE_ENFORCEMENT_VERSION,
                signature_a["prompt_and_code"]["gate_enforcement_version"],
            )
            self.assertIn("runtime", signature_a)
            self.assertIn("configuration", signature_a)
            self.assertIn("medgemma_ecg_core_sha256", signature_a["prompt_and_code"])

    def test_gzip_features_are_discovered_read_and_deduplicated(self) -> None:
        import gzip
        import medgemma_ecg_core as core

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "all_diseases_ecgfeat" / "norm" / "generated"
            split_dir.mkdir(parents=True)
            payload = {
                "global_features": {},
                "interpretation": {},
                "metadata": {"diagnostic_gate": self._pass_gate()},
            }
            gzip_path = split_dir / "001_features.json.gz"
            with gzip.open(gzip_path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)

            plain_duplicate = split_dir / "002_features.json"
            plain_duplicate.write_text(json.dumps(payload), encoding="utf-8")
            with gzip.open(
                split_dir / "002_features.json.gz", "wt", encoding="utf-8"
            ) as handle:
                json.dump(payload, handle)

            samples = list(
                batch_script.iter_samples(
                    root / "all_diseases_ecgfeat",
                    ["generated"],
                )
            )

            self.assertEqual(["001", "002"], [sample.sample_id for sample in samples])
            self.assertEqual(gzip_path, samples[0].features_path)
            self.assertEqual(plain_duplicate, samples[1].features_path)
            self.assertEqual(payload, batch_script._read_feature_json(gzip_path))
            self.assertEqual(payload, core._safe_read_json(gzip_path))

    def test_legacy_process_reads_gzip_and_stops_before_model(self) -> None:
        import gzip

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features_path = root / "003_features.json.gz"
            payload = {
                "global_features": {},
                "interpretation": {},
                "metadata": {
                    "diagnostic_gate": {
                        "state": "stop",
                        "stop_reasons": ["insufficient_beats"],
                        "partial_reasons": [],
                        "allowed_domains": [],
                        "suppressed_domains": ["all"],
                    }
                },
            }
            with gzip.open(features_path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
            sample = batch_script.SampleJob(
                disease_dir_label="norm",
                split_name="generated",
                sample_id="003",
                features_path=features_path,
                report_path=None,
                local_output_dir=root / "local",
                mirror_output_dir=root / "mirror",
            )
            calls: list[str] = []

            result = batch_script.process_sample(
                sample,
                "en",
                "",
                lambda prompt: calls.append(prompt) or "must not run",
            )

            self.assertEqual([], calls)
            self.assertEqual("abstained_diagnostic_gate", result["parse_status"])


if __name__ == "__main__":
    unittest.main()
