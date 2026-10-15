"""Verify release integrity and, optionally, the recomputed paper statistics."""
from pathlib import Path
import argparse
import hashlib
import json

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check-reanalysis", action="store_true")
args = parser.parse_args()
manifest = json.loads((root / "SHA256SUMS.json").read_text())
failures = []
for name, expected in manifest.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit("Release integrity check failed: " + ", ".join(failures))
print(f"Verified {len(manifest)} release files.")
if args.check_reanalysis:
    data = root / "Source_Data_1"
    for actual, reference in (
        ("analysis_recomputed.json", "audit/analysis.json"),
        ("bootstrap_128_recomputed.json", "audit/bootstrap_128.json"),
    ):
        if json.loads((data / actual).read_text()) != json.loads((data / reference).read_text()):
            raise SystemExit(f"Reanalysis differs from the reference: {actual}")
        print(f"Exact JSON result match: {actual}")
