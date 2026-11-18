import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parent
with gzip.open(root/"arms.jsonl.gz","rt") as file:
    records=[json.loads(line) for line in file]
def requests(record):
    batches=[]
    for key,batch in sorted(record["batches"].items()):
        result=[]
        for row in batch["rows"]:
            assert row["cached_tokens"]==0
            assert len(row["tokens"])==48 and len(row["input"]["prompt"])==1024
            assert hashlib.sha256(",".join(map(str,row["tokens"])).encode()).hexdigest()==row["sha256"]
            result.append((row["input"],row["tokens"]))
        batches.append((key,result))
    return batches
baseline={}
for study in ("copy","question","dspine"):
    arms=[r for r in records if r["study"]==study and r["phase"]=="screen"]
    summary=json.loads((root/f"results-{study}.json").read_text())
    assert len(arms)==summary["n_arms"]==12
    baseline[study]=requests(arms[0])
    assert all(r["resident_file_pages_at_start"]==0 for r in arms)
    assert all(requests(r)==baseline[study] for r in arms)==summary["screen_all_tokens_equal"]
    for i,arm in enumerate(arms):
        name=arm["arm"].split("-")
        key=name[1]+":"+name[2]+(":"+name[3][1:] if study=="dspine" else "")
        for n,mode in enumerate(("greedy","seeded")):
            v=median(r["decode_tps"] for r in arm["batches"][f"1024-{n}"]["rows"])
            assert v==summary["rates"][key][mode+"_tps"][i//6]
    print(study,"verified 12 screen arms.", "Matched all served outputs:",summary["screen_all_tokens_equal"])
assert baseline["question"]==baseline["dspine"]
