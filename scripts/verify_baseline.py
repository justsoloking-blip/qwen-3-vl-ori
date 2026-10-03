#!/usr/bin/env python3
"""Check the original tracked snapshot before starting student edits."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", action="store_true", help="Also compare to the official tagged files online")
    args = parser.parse_args()
    source = ROOT / "baseline"
    manifest = json.loads((source / "provenance.json").read_text())
    results = []
    for item in manifest["files"]:
        data = (source / item["path"]).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        row = {"file": item["path"], "sha256": actual, "snapshot_matches": actual == item["sha256"]}
        if args.upstream and "upstream_url" in item:
            with urllib.request.urlopen(item["upstream_url"], timeout=30) as response:
                official = response.read()
            row["upstream_matches"] = official == data
        results.append(row)
    print(json.dumps(results, indent=2))
    if not all(row["snapshot_matches"] and row.get("upstream_matches", True) for row in results):
        raise SystemExit("Snapshot differs. After student edits this is expected; inspect git diff.")


if __name__ == "__main__":
    main()

