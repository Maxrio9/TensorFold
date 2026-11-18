"""Check the published token hashes and reproduce the screen and A-B-B-A timings."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from detail import details
from series import evaluate_run, rate

root = Path(__file__).resolve().parent
result = json.loads((root / "summary.json").read_text())
with gzip.open(root / "arms.jsonl.gz", "rt") as f:
    records = [json.loads(line) for line in f]
arms = {}
tokens = {}
binary_ids = {}
screen = {}
with TemporaryDirectory() as tmp:
    for entry in records:
        name = entry["run"]
        folder = Path(tmp) / name
        folder.mkdir()
        batches = {}
        for key, cell in entry["cells"].items():
            with (folder / (key + "-smi.csv")).open("w", newline="") as f:
                csv.writer(f).writerows(cell["clocks"])
            batch = []
            for lane, request in enumerate(cell["requests"]):
                ids = request["generated_ids"]
                sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
                assert sha == request["token_sha256"]
                identity = entry["streams"], entry["mode"], key, lane
                current = request["request"], ids
                assert tokens.setdefault(identity, current) == current
                batch.append({"output":{"tensorfold":{"time_to_first_token":request["ttft_s"],
                        "tokens_per_second":request["decode_tps"], "token_ids":ids}},
                        "client_submit_monotonic":request["submitted"],"client_done_monotonic":request["finished"]})
            assert len(batch) == entry["streams"]
            batches[key] = batch
        if name.startswith("screen-"):
            _, _, label, threshold = name.split("-")
            assert int(threshold) == entry["threshold"]
            assert binary_ids.setdefault(label, entry["binary_sha256"]) == entry["binary_sha256"]
            screen.setdefault(label + "-" + threshold, []).append(rate(batches["1024-0"], "tokens_per_second"))
        arms[name] = {"batches":batches},folder,entry
    assert screen == result["screen"]
    assert max(("greedy-97","greedy-281"), key=lambda k:mean(screen[k])) == "-".join(map(str,result["selected"]))
    assert sum(k[1]=="correctness" for k in tokens)==196
    for streams, group in result["confirmation"].items():
        for i, record in enumerate(group["rounds"]):
            rows, paths, meta = zip(*(arms[f"confirm-{streams}-{i}-{j}"] for j in range(4)))
            for j, entry in enumerate(meta):
                label, threshold = result["selected"] if j in (1,2) else ("baseline",96)
                assert entry["binary_sha256"] == binary_ids[label] and entry["threshold"] == threshold
            assert details(rows, paths, evaluate_run(rows,paths)) == record
controls = json.loads((root/"conformance.json").read_text())
def checks(side):
    r=controls["conformance-"+side]
    assert all(g["passed"] and not g.get("skipped") for g in r["gates"])
    return [(x["kind"],x["key"],x["sha"]) for x in r["rows"]]
assert checks("baseline")==checks("selected")
print(f"Verified {len(arms)} arms, 196 exact request pairs, six screened settings and all confirmation rounds.")
