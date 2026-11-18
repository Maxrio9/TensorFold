"""Run independent baseline/candidate correctness and fresh-server A-B-B-A trials."""
import argparse
import hashlib
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
from arm import atomic, ROWS, digest

CHANGES=("metadata","normalization","fp8","greedy")
TARGET_PCT=3.0
ROUNDS=3

def runtime(row):
    r=row["output"]
    return r.get("tensorfold") or r["choices"][0]["tensorfold"]

def equality(a,b):
    if set(a["batches"])!=set(b["batches"]): return False
    for key,rows in a["batches"].items():
        other=b["batches"][key]
        if len(rows)!=len(other): return False
        for r,s in zip(rows,other):
            ra,sa=runtime(r),runtime(s)
            if (r["input"],ra["token_ids"],ra["token_sha256"])!=(s["input"],sa["token_ids"],sa["token_sha256"]): return False
    return True

def clocks(folder,key):
    rows=list(csv.reader((folder/(key+"-smi.csv")).open()))
    usable=[]
    for row in rows:
        if len(row)!=11: return None
        try:
            sm,temp,busy=map(float,row[1:4])
            supported,active=(int(v.strip(),0) for v in row[4:6])
        except ValueError: return None
        if not (supported & 0xec) or (active & 0xec): return None
        if any(v.strip().lower()=="active" for v in row[6:]): return None
        if busy >= 20: usable.append(sm)
    return median(usable) if len(usable)>=3 else None

def coverage(r,streams,mode,variant,chip,hashes):
    m=r["meta"]
    if m.get("schema_version")!=2 or (m["mode"],m["streams"],m["variant"],m["compute_capability"])!=(mode,streams,variant,chip): return False
    if any(m[k]!=v for k,v in hashes.items()): return False
    kinds=(0,1,2) if mode=="correctness" and streams==4 else (0,1)
    ns=ROWS if mode=="correctness" else [1024]
    if set(r["batches"])!={f"{n}-{k}" for n in ns for k in kinds}: return False
    for key,rows in r["batches"].items():
        n,k=map(int,key.split("-"))
        if len(rows)!=streams: return False
        for i,row in enumerate(rows):
            ids=runtime(row)["token_ids"]
            req=row["input"]
            if len(req["prompt"])!=n or req["seed"]!=193+i or req["temperature"]!=(0.8 if k==1 or k==2 and i%2 else 0): return False
            if len(ids)!=(16 if mode=="correctness" else 48): return False
            if not all(type(x) is int for x in ids): return False
            token_hash=hashlib.sha256(",".join(str(x) for x in ids).encode()).hexdigest()
            if runtime(row).get("token_sha256")!=token_hash: return False
            usage=row["output"].get("usage",{})
            if usage.get("prompt_tokens")!=n or usage.get("prompt_tokens_details",{}).get("cached_tokens")!=0: return False
    return True

def classify(rounds):
    summary={}
    if not rounds: return summary
    for key in rounds[0]:
        valid=[r[key] for r in rounds if r[key]["clock_qualified"]]
        summary[key]={}
        for metric in ("time_to_first_token","tokens_per_second","aggregate_output_tps_including_prefill"):
            gains=[r[metric]["gain_percent"] for r in valid]
            verdict="insufficient qualified rounds"
            if len(gains)>=ROUNDS:
                mid=median(gains)
                verdict=("material win" if min(gains)>0 and mid>=TARGET_PCT else "material regression" if max(gains)<0 and mid<=-TARGET_PCT else "consistent smaller win" if min(gains)>0 else "consistent smaller regression" if max(gains)<0 else "inconclusive")
            summary[key][metric]={"qualified_rounds":len(gains),"median_gain_percent":median(gains) if gains else None,"verdict":verdict}
    return summary

def rate(rows,metric):
    if metric=="aggregate_output_tps_including_prefill":
        total=sum(len(runtime(row)["token_ids"]) for row in rows)
        wall=max(row["client_done_monotonic"] for row in rows)-min(row["client_submit_monotonic"] for row in rows)
        return total/wall
    return median([float(runtime(row)[metric]) for row in rows])

def evaluate_run(receipts,paths):
    result={}
    for key in receipts[0]["batches"]:
        clock=[clocks(p,key) for p in paths]
        valid=all(x and math.isfinite(x) for x in clock) and max(clock)/min(clock)<=1.01
        part={"sm_clock_mhz":clock,"clock_qualified":bool(valid)}
        for metric in ("time_to_first_token","tokens_per_second","aggregate_output_tps_including_prefill"):
            v=[rate(r["batches"][key],metric) for r in receipts]
            if min(v)<=0: raise ValueError("zero timing")
            a=mean([v[0],v[3]])
            b=mean([v[1],v[2]])
            part[metric]={"base":a,"candidate":b,"gain_percent":100*((a/b if metric=="time_to_first_token" else b/a)-1)}
        result[key]=part
    return result

def run(args):
    root=Path(__file__).resolve().parent
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    def arm(var,label,streams,mode,folder):
        folder=out/var/folder
        dest=folder/"receipt.json"
        binary=args.bin/label/"native/bin/tensorfold-native"
        mark=args.bin/label/"build-ok"
        pin=Path("/manifest/manifest.json")
        if pin.is_file():
            expected_sha=json.loads(pin.read_text())["binaries"][label]["sha256"]
            if digest(binary,"sha256")!=expected_sha: raise ValueError("different binary from manifest")
        elif not mark.is_file(): raise FileNotFoundError(mark)
        model=args.dots if var=="greedy" else args.qwen
        if not dest.exists():
            cmd=[sys.executable,str(root/"arm.py"),str(binary),str(model),str(args.prompts),str(folder),mode,var,str(streams)]
            if args.host_drop: cmd.append("--host-drop")
            subprocess.run(cmd,check=True,timeout=1800)
        r=json.loads(dest.read_text())
        if pin.is_file():
            if r["meta"]["binary_sha256"]!=expected_sha: raise ValueError("receipt does not match pinned binary")
        elif r["meta"]["binary_md5"]!=mark.read_text().split()[0]: raise ValueError("receipt does not match pinned binary")
        chip=args.bin.resolve().name
        if chip not in ("120","121"): chip="121" if Path("/manifest/manifest.json").exists() else None
        hashes={"public_sha256":digest(args.prompts,"sha256"),"arm_sha256":digest(root/"arm.py","sha256"),"model_config_sha256":digest(model/"config.json","sha256")}
        if not coverage(r,streams,mode,var,chip,hashes): raise ValueError("wrong or incomplete arm "+str(folder))
        return r,folder
    results={}
    for var in args.changes:
        results[var]={}
        for stream in (1,4):
            a,_=arm(var,"baseline",stream,"correctness",f"exact-{stream}-A")
            b,_=arm(var,var,stream,"correctness",f"exact-{stream}-B")
            same=equality(a,b)
            results[var][str(stream)]={"tokens_equal":same}
            atomic(out/"status.json",results)
            if not same: break
        if len(results[var])!=2 or not all(v["tokens_equal"] for v in results[var].values()): continue
        if args.only_exact: continue
        for stream in (1,4):
            target=results[var][str(stream)]
            rounds=[]
            target["rounds"]=rounds
            for i in range(ROUNDS):
                seq=[arm(var,label,stream,"timing",f"abba-{stream}-{i}-{j}") for j,label in enumerate(("baseline",var,var,"baseline"))]
                data,paths=zip(*seq)
                if not all(equality(data[0],x) for x in data[1:]):
                    target["timing_token_mismatch"]=True
                    atomic(out/"status.json",results)
                    break
                rounds.append(evaluate_run(data,paths))
                target["verdicts"]=classify(rounds)
                atomic(out/"status.json",results)
    return results
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("bin","qwen","dots","prompts","out"): p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--changes",nargs="+",choices=CHANGES,default=list(CHANGES))
    p.add_argument("--host-drop",action="store_true")
    p.add_argument("--only-exact",action="store_true")
    run(p.parse_args())
