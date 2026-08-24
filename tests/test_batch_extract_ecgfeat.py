from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import batch_extract_ecgfeat as batch_script


class BatchExtractECGFeatTests(unittest.TestCase):
    @staticmethod
    def make_signal(leads: int = 12, points: int = 1000) -> list[list[float]]:
        return [[0.0 for _ in range(points)] for _ in range(leads)]

    @classmethod
    def make_batch(cls, samples: int, leads: int = 12, points: int = 1000) -> list[list[list[float]]]:
        return [cls.make_signal(leads, points) for _ in range(samples)]

    def test_iter_jobs_finds_generated_and_real(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            disease_dir = input_dir / "ami"
            disease_dir.mkdir(parents=True)
            (disease_dir / "generated.pt").write_bytes(b"generated")
            (disease_dir / "real.pt").write_bytes(b"real")
            (disease_dir / "metadata.pt").write_bytes(b"meta")

            jobs = list(batch_script.iter_jobs(input_dir, output_dir, ("generated", "real")))

            self.assertEqual(2, len(jobs))
            self.assertEqual(
                [("ami", "generated"), ("ami", "real")],
                [(job.disease_label, job.source_name) for job in jobs],
            )
            self.assertEqual(output_dir / "ami" / "generated", jobs[0].output_dir)
            self.assertEqual(output_dir / "ami" / "real", jobs[1].output_dir)

    def test_normalize_batch_accepts_single_and_batched_ecg(self) -> None:
        single = self.make_signal()
        batched = self.make_batch(5)
        transposed = [list(col) for col in zip(*single)]

        normalized_single = batch_script.normalize_batch_ecg(single, Path("single.pt"))
        normalized_batched = batch_script.normalize_batch_ecg(batched, Path("batched.pt"))
        normalized_transposed = batch_script.normalize_batch_ecg(transposed, Path("transposed.pt"))

        self.assertEqual((1, 12, 1000), batch_script.infer_shape(normalized_single))
        self.assertEqual((5, 12, 1000), batch_script.infer_shape(normalized_batched))
        self.assertEqual((1, 12, 1000), batch_script.infer_shape(normalized_transposed))

        with self.assertRaises(ValueError):
            batch_script.normalize_batch_ecg([[[[0.0]]]], Path("bad.pt"))

    def test_record_id_is_stable_and_does_not_reveal_disease_label(self) -> None:
        job = batch_script.BatchJob(
            disease_label="atrial_fibrillation",
            source_name="real",
            input_path=Path("/private/input/atrial_fibrillation/real.pt"),
            output_dir=Path("/tmp/output"),
        )

        first = batch_script.opaque_record_id(job, 0)

        self.assertEqual(first, batch_script.opaque_record_id(job, 0))
        self.assertNotIn("atrial", first)
        self.assertNotIn("fibrillation", first)
        self.assertRegex(first, r"^ecg_[0-9a-f]{20}$")

    def test_age_days_overrides_conflicting_year_age_in_batch_metadata(self) -> None:
        header = batch_script.build_sample_header(
            disease_label="normal",
            record_id="opaque",
            source_name="real",
            sampling_rate=500,
            n_points=5000,
            metadata={"age": 40, "age_days": 30, "sex": "female"},
        )

        self.assertEqual(30.0, header["age_days"])
        self.assertAlmostEqual(30.0 / 365.25, header["age"])

    def test_invalid_explicit_age_days_does_not_fall_back_to_year_age(self) -> None:
        header = batch_script.build_sample_header(
            disease_label="normal",
            record_id="opaque",
            source_name="real",
            sampling_rate=500,
            n_points=5000,
            metadata={"age": 70, "age_days": "invalid", "sex": "female"},
        )

        self.assertIsNone(header["age_days"])
        self.assertIsNone(header["age"])

    def test_load_pt_requests_weights_only_deserialization(self) -> None:
        fake_torch = mock.Mock()
        fake_torch.load.return_value = [1.0, 2.0]
        fake_torch.is_tensor.return_value = False

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            payload = batch_script.load_pt(Path("input.pt"))

        self.assertEqual([1.0, 2.0], payload)
        fake_torch.load.assert_called_once_with(
            Path("input.pt"),
            map_location="cpu",
            weights_only=True,
        )

    def test_write_json_supports_gzip_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.json.gz"

            batch_script._write_json(path, {"value": 1.25})

            with gzip.open(path, "rt", encoding="utf-8") as handle:
                self.assertEqual({"value": 1.25}, json.load(handle))

    def test_parser_exposes_performance_and_export_controls(self) -> None:
        args = batch_script.build_arg_parser().parse_args(
            [
                "--workers", "3",
                "--blas-threads", "1",
                "--export-profile", "audit",
                "--no-pt",
                "--gzip-json",
            ]
        )

        self.assertEqual(3, args.workers)
        self.assertEqual(1, args.blas_threads)
        self.assertEqual("audit", args.export_profile)
        self.assertTrue(args.no_pt)
        self.assertTrue(args.gzip_json)

    def test_process_job_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            disease_dir = input_dir / "ami"
            disease_dir.mkdir(parents=True)
            batch_path = disease_dir / "generated.pt"
            batch_path.write_bytes(b"placeholder")
            metadata_path = disease_dir / "metadata.pt"
            metadata_path.write_bytes(b"meta")

            job = batch_script.BatchJob(
                disease_label="ami",
                source_name="generated",
                input_path=batch_path,
                output_dir=output_dir / "ami" / "generated",
                metadata_path=metadata_path,
            )

            saved_pt_paths: list[Path] = []
            report_paths: list[Path] = []

            class FakeExtractor:
                def extract(self, ecg, fs, meta=None):
                    return {
                        "shape": batch_script.infer_shape(ecg),
                        "fs": fs,
                        "meta": None if meta is None else {
                            "age": meta.age,
                            "age_days": meta.age_days,
                            "sex": meta.sex,
                        },
                    }

            def fake_to_dict(result):
                return dict(result)

            def fake_load_pt(path: Path):
                if path == batch_path:
                    return self.make_batch(2)
                if path == metadata_path:
                    return {
                        "label": "AMI",
                        "dataset": "ptbxl",
                        "age": 40,
                        "age_days": 30,
                    }
                raise AssertionError(f"unexpected load path: {path}")

            def fake_save_pt(path: Path, payload) -> None:
                saved_pt_paths.append(path)
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            def fake_report_writer(record_id, hdr, ecg, result, out_path: Path) -> None:
                report_paths.append(out_path)
                shape = batch_script.infer_shape(ecg)
                out_path.write_text(
                    f"{record_id}\n{hdr['dx'][0]}\n{shape[0]}x{shape[1]}",
                    encoding="utf-8",
                )

            summary = batch_script.process_job(
                job=job,
                extractor=FakeExtractor(),
                to_dict_fn=fake_to_dict,
                report_writer=fake_report_writer,
                sampling_rate=100,
                load_pt_fn=fake_load_pt,
                save_pt_fn=fake_save_pt,
                patient_meta_factory=batch_script.PatientMetaStub,
            )

            self.assertEqual(2, summary["processed"])
            self.assertEqual(0, summary["failed"])
            self.assertEqual(2, len(saved_pt_paths))
            self.assertEqual(2, len(report_paths))
            self.assertTrue((job.output_dir / "000_features.json").exists())
            self.assertTrue((job.output_dir / "000_features.pt").exists())
            self.assertTrue((job.output_dir / "000_report.txt").exists())
            self.assertTrue((job.output_dir / "001_features.json").exists())
            self.assertTrue((job.output_dir / "001_features.pt").exists())
            self.assertTrue((job.output_dir / "001_report.txt").exists())
            feature_payload = json.loads(
                (job.output_dir / "000_features.json").read_text(encoding="utf-8")
            )
            self.assertEqual(30.0, feature_payload["meta"]["age_days"])
            self.assertAlmostEqual(30.0 / 365.25, feature_payload["meta"]["age"])

            manifest = json.loads((job.output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("ami", manifest["disease_label"])
            self.assertEqual("generated", manifest["source_name"])
            self.assertEqual(2, manifest["processed"])
            self.assertEqual([], manifest["errors"])

    def test_process_job_writes_annotated_plots_when_writer_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            disease_dir = input_dir / "ami"
            disease_dir.mkdir(parents=True)
            batch_path = disease_dir / "generated.pt"
            batch_path.write_bytes(b"placeholder")

            job = batch_script.BatchJob(
                disease_label="ami",
                source_name="generated",
                input_path=batch_path,
                output_dir=output_dir / "ami" / "generated",
                metadata_path=None,
            )

            annotated_paths: list[Path] = []

            class FakeExtractor:
                def extract(self, ecg, fs, meta=None):
                    return {"shape": batch_script.infer_shape(ecg), "fs": fs}

            def fake_load_pt(path: Path):
                if path == batch_path:
                    return self.make_batch(1)
                raise AssertionError(f"unexpected load path: {path}")

            def fake_save_pt(path: Path, payload) -> None:
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            def fake_report_writer(record_id, hdr, ecg, result, out_path: Path) -> None:
                out_path.write_text(record_id, encoding="utf-8")

            def fake_annotated_writer(record_id, hdr, ecg, result, out_path: Path) -> None:
                annotated_paths.append(out_path)
                out_path.write_bytes(b"png")

            summary = batch_script.process_job(
                job=job,
                extractor=FakeExtractor(),
                to_dict_fn=dict,
                report_writer=fake_report_writer,
                annotated_plot_writer=fake_annotated_writer,
                sampling_rate=100,
                load_pt_fn=fake_load_pt,
                save_pt_fn=fake_save_pt,
                patient_meta_factory=batch_script.PatientMetaStub,
            )

            self.assertEqual(1, summary["processed"])
            self.assertEqual([job.output_dir / "000_ecg_annotated.png"], annotated_paths)
            self.assertTrue((job.output_dir / "000_ecg_annotated.png").exists())

            manifest = json.loads((job.output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(1, manifest["annotated_plots"])

    def test_main_returns_error_when_runtime_dependencies_missing(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(
            batch_script,
            "_import_runtime_components",
            side_effect=RuntimeError("missing module: numpy"),
        ):
            with redirect_stderr(stderr):
                exit_code = batch_script.main([])

        self.assertEqual(1, exit_code)
        self.assertIn("missing module: numpy", stderr.getvalue())

    def test_storage_option_help_describes_actual_artifacts(self) -> None:
        help_text = " ".join(
            batch_script.build_arg_parser().format_help().split()
        )

        self.assertIn("artifacts enabled by the current options", help_text)
        self.assertIn("when enabled, keeps the full detail", help_text)
        self.assertIn("MedGemma batch diagnostics discover", help_text)


if __name__ == "__main__":
    unittest.main()
