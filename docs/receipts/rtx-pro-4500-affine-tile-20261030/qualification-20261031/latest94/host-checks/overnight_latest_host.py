from pathlib import Path
import concurrent.futures,hashlib,json,subprocess,threading,time,uuid
r=Path(__file__).resolve().parent;status={};lock=threading.Lock()
builds=json.loads((r/'overnight-latest-builds.json').read_text());docker=['/usr/bin/docker','-H','unix:///run/codex-ai-bench/docker.sock']
def run(source,row):
 command=row['command'].copy();name='tf4500-build-'+uuid.uuid4().hex[:12];old=command[command.index('--name')+1];command[command.index('--name')+1]=name;command[command.index('--label')+1]='tf4500.overnight.build='+name
 command[-1]='/opt/zig/zig build test-cuda-host test-qwen35-formats-host test-qwen35-draft-trees-host test-qwen35-verify-host test-qwen35-live-host test-qwen35-verify-contract-host test-qwen35-attn-contract-host test-qwen35-launch-args-host -Dsm=120 -Dversion=1.1.6 -Dnvcc=/usr/local/cuda-13.3/bin/nvcc -j4 --summary all'
 with lock:
  reg=r/'overnight-reuse-build-owned-containers.json';a=json.loads(reg.read_text());a.append(name);reg.write_text(json.dumps(a,indent=2))
 start=time.time()
 try:
  with (r/(source+'-host-final.log')).open('w')as log:
   z=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,min(600,1793433550-time.time())))
  binary=r/source/'zig-out/native/bin/tensorfold-native';assert hashlib.sha256(binary.read_bytes()).hexdigest()==row['binary_sha256']
  return source,{'exit_code':z.returncode,'elapsed_s':time.time()-start,'command':command,'native_binary_unchanged':True,'source_head':row['source_head']}
 finally:
  found=subprocess.run([*docker,'inspect',name],capture_output=True,text=True)
  if found.returncode==0:
   assert json.loads(found.stdout)[0]['Config']['Labels']['tf4500.overnight.build']==name
   subprocess.run([*docker,'rm','-f',name],capture_output=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
 for f in concurrent.futures.as_completed([pool.submit(run,k,v)for k,v in builds.items()]):
  k,v=f.result();status[k]=v;(r/'overnight-latest-host-final.json').write_text(json.dumps(status,indent=2)+'\n')
assert all(v['exit_code']==0 for v in status.values())
