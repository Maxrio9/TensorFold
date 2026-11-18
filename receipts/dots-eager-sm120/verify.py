import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,median
root=Path(__file__).resolve().parent
with gzip.open(root/"arms.jsonl.gz","rt") as f:
    records={r["arm"]:r for r in map(json.loads,f)}
review=json.loads((root/"review.json").read_text())
def verify(group,streams,result):
    arms=[records[f"{group}-{i}-s{streams}"] for i in range(4)]
    assert all(a["device_arch"]=="120" and a["resident_file_pages"]==0 for a in arms)
    assert arms[0]["bin_sha256"]==arms[3]["bin_sha256"]!=arms[1]["bin_sha256"]==arms[2]["bin_sha256"]
    assert [a["module_env"]["CUDA_MODULE_LOADING"] for a in arms]==[None,"EAGER","EAGER",None]
    assert [a["module_env"]["CUDA_BINARY_LOADER_THREAD_COUNT"] for a in arms]==[None,"4","4",None]
    assert result["start_to_ready_s"]==[a["startup_s"] for a in arms]
    for key,reported in result["timings"].items():
        times=[];clocks=[];expected=None
        for arm in arms:
            batch=arm["batches"][key];rows=batch["rows"]
            ids=[(r["input"],r["tokens"]) for r in rows]
            if expected is None:expected=ids
            assert ids==expected
            for r in rows:
                assert r["cached_tokens"]==0 and len(r["tokens"])==48 and len(r["input"]["prompt"])==1024
                assert hashlib.sha256(",".join(map(str,r["tokens"])).encode()).hexdigest()==r["sha256"]
            times.append([median(row["ttft_s"] for row in rows),median(row["decode_tps"] for row in rows),sum(len(row["tokens"]) for row in rows)/(max(r["done"] for r in rows)-min(r["sent"] for r in rows))])
            flags=batch["clock_samples"]
            valid=bool(flags) and all(len(c)==11 and int(c[4],0)&0xec and not int(c[5],0)&0xec and not any(v.strip().lower()=="active" for v in c[6:]) for c in flags)
            sm=[float(c[1]) for c in flags if len(c)==11 and float(c[3])>=20]
            clocks.append(median(sm) if valid and len(sm)>=3 else None)
        assert clocks==reported["sm_clock_mhz"]
        assert reported["clock_qualified"]==bool(all(clocks) and max(clocks)/min(clocks)<=1.01)
        for i,metric in enumerate(("time_to_first_token","tokens_per_second","aggregate_output_tps_including_prefill")):
            a,b=mean([times[0][i],times[3][i]]),mean([times[1][i],times[2][i]])
            gain=100*((a/b if i==0 else b/a)-1)
            assert math.isclose(gain,reported[metric]["gain_percent"],abs_tol=1e-9)
    assert result["exact"]
verify("screen",1,review["screen"])
for streams,rounds in review.get("confirmation",{}).items():
    for i,r in enumerate(rounds):verify("confirm"+str(i),int(streams),r)
print("Verified cold files, timing inputs, generated IDs, startup times, paired gains and clock qualification.")
