"""A fixed, bounded dots.mocr experiment using two published binaries and three tile settings."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from statistics import mean,median
from arm import atomic,digest
from detail import details
from series import coverage,equality,evaluate_run,rate,classify
configs=(("baseline",96),("baseline",97),("baseline",281),("greedy",96),("greedy",97),("greedy",281))

def has_coverage(report):
    return (report.get("upstream_conformance") is True and
            all(sum(row[k]["clock_qualified"] for row in report["confirmation"][n]["rounds"]) >= 3
                for n in ("1","4") for k in ("1024-0","1024-1")))

def experiment(out, bins, model, support, chip, execute, hashes, stop):
    out.mkdir(parents=True,exist_ok=True)
    arm=support/"arm-fast.py"
    fixture=support/"bench_prompts.json"
    screening={}
    reference={}
    arm_hash=digest(arm,"sha256")
    public_hash=digest(fixture,"sha256")
    config_hash=digest(model/"config.json","sha256")
    device=None
    def run(key,config,streams,mode):
        nonlocal device
        binary=bins/config[0]/"native/bin/tensorfold-native"
        folder=out/key
        receipt=folder/"receipt.json"
        if not receipt.exists():
            if time.monotonic()>stop-30: raise TimeoutError("campaign deadline")
            folder.mkdir(parents=True,exist_ok=True)
            args=[sys.executable,str(arm),str(binary),str(model),str(fixture),str(folder),mode,"greedy",str(streams)]
            if chip=="120" and mode=="timing":args.append("--host-drop")
            execute(args,config[1],folder,stop)
        row=json.loads(receipt.read_text())
        assert (folder/"threshold.txt").read_text().strip()==str(config[1]),key
        assert coverage(row,streams,mode,"greedy",chip,{"binary_sha256":hashes[config[0]],"arm_sha256":arm_hash,"public_sha256":public_hash,"model_config_sha256":config_hash}),key
        if device is None: device=row["meta"]["device"]
        assert row["meta"]["device"]==device
        if mode=="timing":assert not any(row["meta"]["file_resident_pages_before_start"].values())
        ref=reference.setdefault((streams,mode),row)
        assert equality(row,ref),("input or token mismatch",key)
        return row,folder
    for idx,config in enumerate((*configs,*reversed(configs))):
        key=f"screen-{idx:02d}-{config[0]}-{config[1]}"
        row,_=run(key,config,1,"timing")
        screening.setdefault(f"{config[0]}-{config[1]}",[]).append(rate(row["batches"]["1024-0"],"tokens_per_second"))
        atomic(out/"screen.json",screening)
    chosen=max(configs[-2:],key=lambda cfg:mean(screening[f"{cfg[0]}-{cfg[1]}"]))
    atomic(out/"selection.json",{"chosen":chosen,"screen_decode_tps":screening})
    for streams in (1,4):
        for side,config in (("A",configs[0]),("B",chosen)):
            run(f"exact-{streams}-{side}",config,streams,"correctness")
    report={"selected":chosen,"screen":screening,"device":device,"confirmation":{}}
    for streams in (1,4):
        rounds=[]
        for idx in range(5):
            pairs=[run(f"confirm-{streams}-{idx}-{j}",cfg,streams,"timing") for j,cfg in enumerate((configs[0],chosen,chosen,configs[0]))]
            rows,paths=zip(*pairs)
            rounds.append(details(rows,paths,evaluate_run(rows,paths)))
            keys=rounds[0].keys()
            qualified={k:[x[k] for x in rounds if x[k]["clock_qualified"]][:3] for k in keys}
            cells={}
            for k,valid in qualified.items():
                cells[k]=classify([{k:x} for x in valid]).get(k,{})
            report["confirmation"][str(streams)]={"rounds":rounds,"selected":cells}
            atomic(out/"combination-results.json",report)
            if all(len(v)==3 for v in qualified.values()):break
    controls=[]
    for label,config in (("baseline",configs[0]),("selected",chosen)):
        directory=out/("conformance-"+label);directory.mkdir(exist_ok=True)
        receipt=directory/"receipt.json"
        binary=bins/config[0]/"native/bin/tensorfold-native"
        if not receipt.exists():
            execute([sys.executable,str(support/"run_conformance.py"),str(support/"dots-profile.json"),"--binary",str(binary),"--model-dir",str(model),"--gates","drafted,repeat,concurrent","--output",str(receipt)],config[1],directory,stop)
        data=json.loads(receipt.read_text())
        assert data["binary_sha256"]==hashes[config[0]] and all(g["passed"] and not g.get("skipped") for g in data["gates"])
        controls.append([(x["kind"],x["key"],x["sha"]) for x in data["rows"]])
    assert controls[0]==controls[1] and all(x[-1] for x in controls[0])
    report["upstream_conformance"]=True
    report["evidence_complete"]=has_coverage(report)
    atomic(out/"combination-results.json",report)
    return report
