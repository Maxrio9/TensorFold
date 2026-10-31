from pathlib import Path
import importlib.util,json,subprocess,time,urllib.request
r=Path(__file__).resolve().parent;o=r/'maintainer-noise-results';o.mkdir(exist_ok=True);plan=json.loads((r.parent/'submission/maintainer-followup/noise-plan.json').read_text());deadline=plan['deadline_epoch']
spec=importlib.util.spec_from_file_location('reserve','/shared-cache');reserve=importlib.util.module_from_spec(spec);spec.loader.exec_module(reserve)
def profile_now():
 for attempt in range(3):
  try:return reserve.policy()
  except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
   if attempt==2:raise
   time.sleep(.1)

while time.time()<deadline-120:
 try:ok=json.load(urllib.request.urlopen('http://127.0.0.1:19092/resources',timeout=5))['can_request_background']
 except OSError:ok=False
 if not ok:time.sleep(5);continue
 profile=profile_now();assert profile['reserve_gib']==0,'Headless qualification profile changed; no silent reserve substitution'
 cmd=json.loads((r/'overnight-latest-command.json').read_text())['command'];cmd[cmd.index('--timeout')+1]=str(min(2200,int(deadline-time.time())-10));old=str(r/'overnight-latest-results')+':/task/results';cmd[cmd.index(old)]=str(o)+':/task/results';cmd[-1]='/task/maintainer_noise_worker.py';(o/'command.json').write_text(json.dumps({'command':cmd,'profile':profile,'plan':plan},indent=2)+'\n')
 with(o/'guard.log').open('a')as log,(o/'telemetry.csv').open('a')as telemetry:
  monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used,temperature.gpu','--format=csv','-l','1'],stdout=telemetry);p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);changed=False
  try:
   while p.poll()is None:
    now=profile_now()
    if any(now[k]!=profile[k]for k in['compute_mode','graphical_desktop','reserve_gib']):changed=True;p.terminate();break
    time.sleep(3)
   code=p.wait(timeout=30)
  finally:
   if p.poll()is None:p.terminate();p.wait(timeout=30)
   monitor.terminate();monitor.wait(timeout=10)
 (o/'exit.json').write_text(json.dumps({'exit_code':code,'profile_changed':changed},indent=2)+'\n')
 if code==75 and not changed:time.sleep(5);continue
 raise SystemExit(code)
raise SystemExit('Background noise-check deadline reached')
