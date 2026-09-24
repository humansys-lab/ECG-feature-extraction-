#!/usr/bin/env python3
"""Post-install smoke test for released artifacts (document 07, steps 6, 8 and 11).

Run in a clean environment where ecg-records (optionally with its ``viz``
extra) was installed from wheels, TestPyPI or PyPI.  Checks the
import, the packaged schema resources, canonical encode/decode, the reported
package/schema versions, and one representative extraction of a synthetic
12-lead signal (no repository files needed).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from importlib import resources


def synthetic_signal():
    import numpy as np

    t = np.arange(5000) / 500
    signal = np.zeros((12, t.size))
    for index in range(12):
        for r in 0.5 + np.arange(10):
            signal[index] += (1.0 + 0.1 * index) * np.exp(-0.5 * ((t - r) / 0.011) ** 2)
            signal[index] += 0.3 * np.exp(-0.5 * ((t - r - 0.3) / 0.05) ** 2)
            signal[index] += 0.12 * np.exp(-0.5 * ((t - r + 0.16) / 0.022) ** 2)
    return signal + np.random.default_rng(0).normal(0, 0.005, signal.shape)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="expected ecg-records version")
    parser.add_argument("--with-viz", action="store_true", help="also plot with ecgfeat.viz (needs the viz extra)")
    args = parser.parse_args(argv)

    import ecgfeat
    from ecgfeat import dumps_record, ecg_record, loads_record, query_measurement

    assert importlib.metadata.version("ecg-records") == args.version == ecgfeat.__version__, "version mismatch"
    schema = json.loads(resources.files("ecgfeat").joinpath("schemas", "ecg-record", "1.0", "schema.json").read_text())
    registry = json.loads(resources.files("ecgfeat").joinpath("schemas", "ecg-record", "1.0",
                                                              "validation-evidence.json").read_text())
    assert schema["$id"] == "urn:ecg-record:1.0" and registry["fields"], "schema resources missing"
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    signal = synthetic_signal()
    record = ecg_record(signal, sampling_rate=500, lead_names=leads)
    data = dumps_record(record)
    assert dumps_record(loads_record(data)) == data, "canonical round trip failed"
    assert record.schema_version == "1.0.0" and record.provenance["library_version"] == args.version
    assert len(data) <= 24_000, f"summary {len(data)} bytes"
    heart_rate = query_measurement(record, "heart_rate_bpm").value
    report = {"ecg-records": args.version, "schema": record.schema_version, "summary_bytes": len(data),
              "beats": len(record.axes["beats"]), "heart_rate_bpm": heart_rate}
    if args.with_viz:
        import matplotlib

        from ecgfeat.viz import plot_record

        matplotlib.use("Agg")
        figure, axes = plot_record(signal, record, leads=["II"])
        report.update(matplotlib=matplotlib.__version__, plotted_axes=len(axes))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
