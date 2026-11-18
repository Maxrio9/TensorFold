"""Run one isolated, cold TensorFold server arm. A new process and receipt per arm."""
import argparse
import concurrent.futures
import ctypes
import hashlib
import http.client
import json
import math
import mmap
import os
from pathlib import Path
import socket
import signal
import subprocess
import threading
import time

ROWS = [1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1024]
THROTTLE = ["sw_power_cap", "hw_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown", "sw_thermal_slowdown"]
QUERY = ["timestamp","clocks.sm","temperature.gpu","utilization.gpu","clocks_event_reasons.supported","clocks_event_reasons.active"] + ["clocks_event_reasons."+x for x in THROTTLE]
libc = ctypes.CDLL(None, use_errno=True)

def digest(p, algorithm):
    with p.open("rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()

def evict(p):
    with p.open("rb") as f:
        size = os.fstat(f.fileno()).st_size
        if not size:
            return 0
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
            first = ctypes.c_char.from_buffer(mapping)
            vec = (ctypes.c_ubyte * math.ceil(size/mmap.PAGESIZE))()
            rc = libc.mincore(ctypes.c_void_p(ctypes.addressof(first)),ctypes.c_size_t(size),vec)
            del first
            if rc: raise OSError(ctypes.get_errno(),"mincore")
            return sum(x & 1 for x in vec)

def post(port, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1",port,timeout=240)
    try:
        c.request("GET" if body is None else "POST",path,None if body is None else json.dumps(body),{"Content-Type":"application/json"})
        r = c.getresponse()
        raw = r.read()
        if r.status != 200: raise RuntimeError(f"{path} HTTP {r.status} {raw[:500]!r}")
        return json.loads(raw)
    finally: c.close()

def make_prompts(port,public):
    source=public.read_text()
    profile=json.loads((public.parent/"tools/conformance_profiles/qwen35-cuda.json").read_text())
    output=[]
    for i,key in enumerate(["sky","poem","primes","count"]):
        middle=len(source)*i//4
        prefix=(source[middle:]+source[:middle])[:18000]
        head=post(port,"/v1/tokenize",{"prompt":prefix,"add_special_tokens":False})["tokens"]
        tail=post(port,"/v1/tokenize",{"prompt":"\n\nQuestion: "+profile["prompts"][key]+"\nAnswer:","add_special_tokens":False})["tokens"]
        assert len(head)>=1024 and 0<len(tail)<128
        output.append(head[:1024-len(tail)]+tail)
    return output

def request_batch(port, prompts, rows, modes, n, seed):
    barrier=threading.Barrier(len(modes))
    def query(i):
        req={"model":"bench","prompt":prompts[i][:rows],"max_tokens":n,"temperature":0.8 if modes[i] else 0.0,
             "seed":seed+i,"top_p":0.95 if modes[i] else 1.0,"ignore_eos":True,"return_token_ids":True,"draft":True}
        barrier.wait(timeout=20)
        sent=time.perf_counter()
        reply=post(port,"/v1/completions",req)
        received=time.perf_counter()
        runtime=reply.get("tensorfold") or reply["choices"][0].get("tensorfold",{})
        ids=runtime.get("token_ids")
        if not isinstance(ids,list) or len(ids)!=n or not all(type(x) is int for x in ids): raise RuntimeError(f"unqualified token IDs: {reply!r}"[:900])
        expected=hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()
        if runtime.get("token_sha256")!=expected or reply["usage"]["prompt_tokens"]!=rows: raise RuntimeError("missing token hash or incorrect prompt count")
        if reply["usage"].get("prompt_tokens_details",{}).get("cached_tokens") != 0: raise RuntimeError("unexpected cache hit")
        return {"input":req,"output":reply,"client_submit_monotonic":sent,"client_done_monotonic":received}
    with concurrent.futures.ThreadPoolExecutor(len(modes)) as pool:
        return list(pool.map(query,range(len(modes))))

def atomic(p,data):
    temp=p.with_suffix(".tmp")
    temp.write_text(json.dumps(data, separators=(",",":"),allow_nan=False)+"\n")
    temp.replace(p)

def trace_server_pid(binary, port):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal(): continue
        try:
            args=(entry/"cmdline").read_bytes().split(b"\0")
            if os.readlink(entry/"exe")==str(binary) and b"--port" in args and args[args.index(b"--port")+1]==str(port).encode():
                return int(entry.name)
        except (OSError,IndexError): continue
    return None

def main():
    args=argparse.ArgumentParser()
    args.add_argument("binary",type=Path)
    args.add_argument("model",type=Path)
    args.add_argument("public",type=Path)
    args.add_argument("out",type=Path)
    args.add_argument("mode",choices=["correctness","timing","trace"])
    args.add_argument("variant",choices=["metadata","normalization","fp8","greedy"])
    args.add_argument("streams",type=int,choices=[1,4])
    args.add_argument("--host-drop",action="store_true")
    args.add_argument("--drafter",required=True,type=Path)
    a=args.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    binary=a.binary.resolve()
    if (a.out/"receipt.json").exists(): raise RuntimeError("completed arm already exists")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port=sock.getsockname()[1]
    device=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,compute_cap,driver_version,clocks.max.sm,power.limit","--format=csv,noheader"],text=True).strip()
    target=next((i for i in binary.parts if i in ("120","121")),None)
    actual=subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=compute_cap","--format=csv,noheader,nounits"],text=True).strip().replace(".","")
    if target is None or actual!=target: raise ValueError(f"wrong physical GPU {actual} for {target}")
    meta={"schema_version":2,"arm_sha256":digest(Path(__file__),"sha256"),"compute_capability":actual,"device":device,"mode":a.mode,"variant":a.variant,"streams":a.streams,"binary_md5":digest(binary,"md5"),
          "binary_sha256":digest(binary,"sha256"),"public_sha256":digest(a.public,"sha256"),"model":str(a.model),
          "clocks":"as configured, no clock changes","max_tokens":48}
    cmd=[str(binary),"serve",str(a.model),"--backend","cuda","--name","bench","--host","127.0.0.1","--port",str(port),
         "--context","4096","--parallel",str(a.streams),"--drafter",str(a.drafter)]
    cmd.append("--no-update-check")
    if a.variant!="greedy": cmd+=["--no-thinking","--kv-cache","fp8" if a.variant=="fp8" else "bf16","--prompt-cache-gib","0"]
    if a.mode=="trace":
        cmd=["nsys","profile","-t","cuda","--cuda-graph-trace=node","--sample=none","--cpuctxsw=none","-o",str(a.out/"kernels"),*cmd]
    meta["argv"]=cmd
    meta["model_config_sha256"]=digest(a.model/"config.json","sha256")
    meta["drafter_window"]=os.getenv("TF_DSPINE_WINDOW","checkpoint default")
    meta["draft_nodes"]=os.getenv("TF_DRAFT_NODES","default=15")
    meta["drafter_config_sha256"]=digest(a.drafter/"config.json","sha256")
    if a.host_drop:
        subprocess.run(["sudo","-n","sh","-c","sync && echo 3 > /proc/sys/vm/drop_caches"],check=True)
    files=sorted(p for p in a.model.iterdir() if p.is_file() and p.suffix in (".safetensors",".json",".jinja",".txt"))
    checked={p.name:evict(p) for p in [binary,*files,*sorted(a.drafter.glob("*.safetensors")),a.drafter/"config.json"]}
    meta["file_resident_pages_before_start"]=checked
    meta["host_cache_drop"]=a.host_drop
    if a.mode=="timing" and any(checked.values()): raise RuntimeError(f"cold file pages still resident: {checked}")
    with (a.out/"server.log").open("w") as log:
        run_env=dict(os.environ)
        if a.mode=="trace": run_env["TMPDIR"]=str(a.out.resolve())
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=run_env)
        profiled_pid=None
        try:
            until=time.monotonic()+360
            while time.monotonic()<until:
                if proc.poll() is not None: raise RuntimeError("server exited during startup; see server.log")
                try:
                    if any(x["id"]=="bench" for x in post(port,"/v1/models")["data"]): break
                except (ConnectionError,OSError,http.client.HTTPException): pass
                time.sleep(.4)
            else: raise RuntimeError("server startup timed out")
            if a.mode=="trace":
                profiled_pid=trace_server_pid(binary,port)
                if profiled_pid is None: raise RuntimeError("cannot identify profiled server")
            prompts=make_prompts(port,a.public)
            batches={}
            for rows in (ROWS if a.mode=="correctness" else [1024]):
                modes=([0]*a.streams,[1]*a.streams)
                if a.mode=="trace": modes=modes[:1]
                if a.mode=="correctness" and a.streams==4: modes=(*modes,[0,1,0,1])
                for which,mode in enumerate(modes):
                    key=f"{rows}-{which}"
                    with (a.out/(key+"-smi.csv")).open("w") as csvout:
                        probe=subprocess.Popen(["nvidia-smi","-i","0","--query-gpu="+",".join(QUERY),"--format=csv,noheader,nounits","-lms","25"],stdout=csvout,stderr=subprocess.STDOUT)
                        try: batches[key]=request_batch(port,prompts,rows,mode,48 if a.mode!="correctness" else 16,193)
                        finally:
                            if probe.poll() is None: probe.terminate()
                            probe.wait(timeout=10)
                atomic(a.out/"partial.json",batches)
            atomic(a.out/"receipt.json",{"meta":meta,"batches":batches})
        finally:
            if proc.poll() is None:
                if profiled_pid is not None and trace_server_pid(binary,port)==profiled_pid:
                    os.kill(profiled_pid,signal.SIGTERM)
                else: proc.terminate()
                try: proc.wait(timeout=120 if profiled_pid else 60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
if __name__=="__main__": main()
