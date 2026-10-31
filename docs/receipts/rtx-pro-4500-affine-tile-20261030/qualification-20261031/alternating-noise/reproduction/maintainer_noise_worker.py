"""Six predeclared independent ABA triplets; canonical greedy code/prose, seven repeats per boot."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,http.client,json,os,subprocess,time,traceback
R=Path('/task');O=R/'results';O.mkdir(exist_ok=True);source='overnight-latest-tile';binary=Path('/submission')/source/'zig-out/native/bin/tensorfold-native';expected=json.loads((R/'overnight-latest-binaries.json').read_text())[source];assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected
state=json.loads((O/'state.json').read_text())if(O/'state.json').exists()else{'complete':False,'boots':[],'protocol':'Six independently calibrated stock/candidate/stock triplets, all18boots predefined. Greedy code/prose256tokens, seven seeded repetitions plus warmup per prompt. Boot is unit; no per-token/repetition pseudoreplication. All attempts retained. Native binary unchanged. Full serving features and131072context retained.','started_utc':datetime.now(timezone.utc).isoformat()};server=None;log=None;reference=None

def save():(O/'state.json').write_text(json.dumps(state,indent=2)+'\n')
def get(path):
 c=http.client.HTTPConnection('127.0.0.1',18080,timeout=10)
 try:
  c.request('GET',path);z=c.getresponse();assert z.status==200;return json.load(z)
 finally:c.close()
try:
 for index in range(18):
  flag='1'if index%3==1 else'0';name=f'{index:02d}-t{index//3}-{flag}';previous=next((b for b in state['boots']if b['name']==name and b.get('complete')),None)
  if previous:
   rows=json.loads((O/(name+'.json')).read_text());hashes={x['prompt']:[e['token_sha']for e in x['evidence']]for x in rows}
   if reference is None:reference=hashes
   assert reference==hashes;continue
  state['boots']=[b for b in state['boots']if b['name']!=name];boot={'name':name,'triplet':index//3,'override':flag,'source':source,'binary_sha256':expected,'start_utc':datetime.now(timezone.utc).isoformat()};state['boots'].append(boot);save()
  env={**os.environ,'TF_SRC':'/submission/'+source,'MODELS':'/models','HOST':'127.0.0.1','PORT':'18080','CONTEXT':'131072','PARALLEL':'1','NAME':'qwen-community','ALIAS':'qwen-community-alias','PROMPT_CACHE_GIB':'8','KV_CACHE':'fp8','PROMPT_PRECISION':'int8','TENSORFOLD_MEMORY_RESERVE_GIB':'0.5','TENSORFOLD_CUDA_KERNELS':'native','HOME':'/tmp/maintainer-noise-'+name,'MAX_TOKENS':'2048','TF_CHECKPOINT_REVISION':'22d8d538154e0e5b6f8dcb1fcc74a60a0798104c','TF_CUDA_TEST_MLP128':flag}
  log=(O/(name+'-server.log')).open('w');server=subprocess.Popen(['/bin/bash','/recipe/serve.sh'],env=env,stdout=log,stderr=log);deadline=time.monotonic()+300
  while time.monotonic()<deadline:
   assert server.poll()is None,'Server exited'
   try:
    health=get('/health')
    if not health.get('warming'):break
   except OSError:pass
   time.sleep(.5)
  else:raise RuntimeError('Boot timeout')
  assert health['max_batch_size']==1 and health['version']=='1.1.6';boot['health']=health;boot['manifest']=get('/v1/benchmark/manifest');assert all(x['context_length']==131072 for x in get('/v1/models')['data']);save()
  cmd=['python3','/submission/base/tools/bench_openai.py','http://127.0.0.1:18080','qwen-community','--tokens','256','--reps','7','--temperatures','0','--output',str(O/(name+'.json'))];boot['command']=cmd
  with(O/(name+'-client.log')).open('w')as f:assert subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode==0
  rows=json.loads((O/(name+'.json')).read_text());assert len(rows)==2
  for row in rows:
   assert len(row['evidence'])==7
   assert all(e['complete']and e['tokens']==256 and e['cached_tokens']==0 for e in row['evidence'])
  hashes={x['prompt']:[e['token_sha']for e in x['evidence']]for x in rows}
  if reference is None:reference=hashes
  assert hashes==reference,'Greedy output mismatch'
  marker=f'experimental MLP128 override={flag}, enabled={str(flag=="1").lower()}';assert marker in(O/(name+'-server.log')).read_text();boot['override_startup_confirmed']=True;boot['token_parity']=True
  boot['final_health']=get('/health');boot['memory']={k:int(v.split()[0])*1024 for line in Path(f'/proc/{server.pid}/status').read_text().splitlines()if ':'in line for k,v in[line.split(':',1)]if k in['VmHWM','VmRSS']}
  server.terminate();server.wait(timeout=20);server=None;log.close();log=None;boot['complete']=True;boot['end_utc']=datetime.now(timezone.utc).isoformat();save();time.sleep(2)
 state['complete']=True
except Exception:state['error']=traceback.format_exc()
finally:
 if server is not None:
  server.terminate()
  try:server.wait(timeout=20)
  except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
 if log:log.close()
 state['ended_utc']=datetime.now(timezone.utc).isoformat();save()
if not state['complete']:raise SystemExit(1)
