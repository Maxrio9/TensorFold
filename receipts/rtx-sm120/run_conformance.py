"""Use upstream conformance gates with a CLI-compatible cache-off dots server."""
import signal
import os
import subprocess
import sys
from pathlib import Path
tools=Path(__file__).resolve().parent.parent/"TensorFold/tools"
if tools.is_dir(): sys.path.insert(0,str(tools))
import conformance
if os.getenv("DESKTOP_ADMISSION_LEASE"):
    # The desktop guard owns this process group and must be able to reap the server.
    class AdmittedSubprocess:
        def __getattr__(self,name): return getattr(subprocess,name)
        def Popen(self,*args,**kwargs):
            kwargs["start_new_session"]=False
            return subprocess.Popen(*args,**kwargs)
    conformance.subprocess=AdmittedSubprocess()
original_enter=conformance.Server.__enter__
def enter(server):
    try: return original_enter(server)
    except BaseException:
        if hasattr(server,"log"): server.__exit__()
        raise
conformance.Server.__enter__=enter

upstream=conformance.Server.command
def command(server,port):
    cmd=upstream(server,port)
    if server.profile.get("no_cache_option"):
        i=cmd.index("--prompt-cache-gib")
        assert cmd[i+1]=="0"
        del cmd[i:i+2]
    return cmd

def terminate(signum,frame): raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,terminate)
conformance.Server.command=command
sys.exit(conformance.main())
