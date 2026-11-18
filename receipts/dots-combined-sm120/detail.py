"""Preserve device telemetry coverage and each concurrent request's paired gain."""
import csv,math
from statistics import mean
from pathlib import Path
from series import runtime

def inspect_clocks(path):
    values=[]
    reasons=[]
    for row in csv.reader(Path(path).read_text().splitlines()):
        try:
            if len(row)!=11 or int(row[4],0)&0xec != 0xec or any(x.strip().lower() not in ("active","not active") for x in row[6:]):
                reasons.append("unsupported clock-event data");continue
            if int(row[5],0)&0xec or any(x.strip().lower()=="active" for x in row[6:]):
                reasons.append("power, thermal or hardware clock event")
            mhz,busy=map(float,(row[1],row[3]))
            if not math.isfinite(mhz) or not math.isfinite(busy):reasons.append("nonfinite telemetry")
            elif busy>=20:values.append(mhz)
        except ValueError:reasons.append("malformed telemetry")
    if len(values)<3:reasons.append("fewer than three busy samples")
    return {"min":min(values,default=None),"max":max(values,default=None),"count":len(values),"reasons":sorted(set(reasons))}

def details(rows,paths,metrics):
    for key,m in metrics.items():
        clocks=[inspect_clocks(p/(key+"-smi.csv")) for p in paths]
        m["telemetry"]=clocks
        m["median_clock_qualified"]=m["clock_qualified"]
        m["clock_qualified"]=m["clock_qualified"] and not any(x["reasons"] for x in clocks) and max(x["max"] for x in clocks)<=1.01*min(x["min"] for x in clocks)
        lanes=[]
        for batch in zip(*(r["batches"][key] for r in rows)):
            lane={}
            for metric in ("time_to_first_token","tokens_per_second"):
                vals=[float(runtime(x)[metric]) for x in batch]
                a,b=mean(vals[::3]),mean(vals[1:3])
                lane[metric]={"baseline":a,"candidate":b,"gain_percent":100*((a/b if metric=="time_to_first_token" else b/a)-1)}
            lanes.append(lane)
        m["lane_metrics"]=lanes
    return metrics
