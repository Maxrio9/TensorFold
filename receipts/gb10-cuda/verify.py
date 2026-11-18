"""Recompute paired token equality and three-round gains from the public arm records."""
from pathlib import Path
from statistics import mean, median
import csv
import gzip
import hashlib
import json
import math

here = Path(__file__).resolve().parent
summary = json.loads((here / "summary.json").read_text())
binary = json.loads((here / "binary-sha256.json").read_text())
arms = {}
with gzip.open(here / "arms.jsonl.gz", "rt") as data:
    for line in data:
        r = json.loads(line)
        variant, run = r["variant"], r["run"]
        assert (variant, run) not in arms
        arms[variant, run] = r
        for cell in r["cells"].values():
            for row in cell["requests"]:
                text = ",".join(map(str, row["generated_ids"]))
                assert hashlib.sha256(text.encode()).hexdigest() == row["token_sha256"]
                assert row["cached_tokens"] == 0 and len(row["request"]["prompt"]) == row["prompt_tokens"]
                assert len(row["generated_ids"]) == row["request"]["max_tokens"]

def signature(cell):
    return [(r["request"], r["generated_ids"]) for r in cell["requests"]]

def clock(rows):
    busy = []
    for r in rows:
        if len(r) != 11 or not int(r[4], 0) & 0xec or int(r[5], 0) & 0xec:
            return None
        if any(x.strip().lower() == "active" for x in r[6:]):
            return None
        if float(r[3]) >= 20:
            busy.append(float(r[1]))
    return median(busy) if len(busy) >= 3 else None

for variant, groups in summary.items():
    count = 0
    for n, group in groups.items():
        a, b = [arms[variant, f"exact-{n}-{side}"] for side in "AB"]
        assert [a["binary_sha256"], b["binary_sha256"]] == [binary["baseline"], binary[variant]]
        for key, cell in a["cells"].items():
            assert signature(cell) == signature(b["cells"][key])
            count += len(cell["requests"])
        for key, reported in group["timing"].items():
            gains = []
            reference = None
            for attempt in group["attempts"]:
                kind, i = attempt["round"].split("-")
                records = [arms[variant, f"{kind}-{n}-{i}-{j}"] for j in range(4)]
                assert [r["binary_sha256"] for r in records] == [binary[x] for x in ("baseline", variant, variant, "baseline")]
                cells = [r["cells"][key] for r in records]
                reference = reference or signature(cells[0])
                assert all(signature(c) == reference for c in cells)
                mhz = [clock(c["clocks"]) for c in cells]
                valid = all(mhz) and max(mhz) / min(mhz) <= 1.01
                assert bool(valid) == attempt["results"][key]["clock_qualified"]
                if not valid:
                    continue
                results = {}
                for m, field in (("tokens_per_second", "decode_tps"), ("time_to_first_token", "ttft_s"), ("aggregate_output_tps_including_prefill", None)):
                    rates = []
                    for c in cells:
                        rows = c["requests"]
                        if field:
                            rates.append(median(r[field] for r in rows))
                        else:
                            elapsed = max(r["finished"] for r in rows) - min(r["submitted"] for r in rows)
                            rates.append(sum(len(r["generated_ids"]) for r in rows) / elapsed)
                    base, patched = mean(rates[::3]), mean(rates[1:3])
                    gain = 100 * ((base / patched if field == "ttft_s" else patched / base) - 1)
                    assert math.isclose(gain, attempt["results"][key][m]["gain_percent"], abs_tol=1e-8)
                    results[m] = gain
                gains.append((attempt["round"], results))
            assert [g[0] for g in gains[:3]] == reported["selected"]
            for m, result in reported["metrics"].items():
                assert math.isclose(median(g[1][m] for g in gains[:3]), result["median_gain_percent"], abs_tol=1e-8)
    assert count == 196
    print(f"{variant}: {count} paired exact requests; timed inputs, tokens, clocks and gains verified")
