from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import batch_medgemma_diagnostics as batch_script


class BatchMedGemmaDiagnosticsTests(unittest.TestCase):
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
                        "metadata": {},
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
                        "metadata": {},
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


if __name__ == "__main__":
    unittest.main()
