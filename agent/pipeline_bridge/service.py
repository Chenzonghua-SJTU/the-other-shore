"""Owned offline PySR service; initialize before requesting mission."""
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import sys
import threading
import time
from .bootstrap import repository
repository()
from ..repository.shore_agent.symbolic_service import WorkerPool, make_server


@contextmanager
def symbolic_service(output_dir, *, size=1, timeout=900., external_endpoint=None):
    if external_endpoint:
        yield external_endpoint
        return
    agent_dir = Path(__file__).resolve().parents[1]
    frozen = Path(output_dir).resolve()/'submission/agent'
    if frozen.is_dir():agent_dir=frozen
    vendor = agent_dir/'vendor_runtime/pysr'
    output = Path(output_dir).resolve()/'pysr-service'
    output.mkdir(parents=True,exist_ok=True)
    project = output/'julia-project'
    project.mkdir(exist_ok=True)
    for name in ('Project.toml','Manifest.toml'):
        shutil.copy2(vendor/'julia_env'/name,project/name)
    env = os.environ.copy()
    env.pop('RIFT_SESSION_TOKEN',None)
    env.pop('RIFT_CONSOLE_URL',None)
    import rift as official_sdk
    sdk = os.environ.get('SHORE_SDK_ROOT',str(Path(official_sdk.__file__).resolve().parent.parent))
    site = vendor/('lib/python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor)+'/site-packages')
    if not site.is_dir():raise RuntimeError('Bundled PySR requires the official Linux CPython 3.12 runtime')
    env.pop('PYTHONHOME',None);env.pop('VIRTUAL_ENV',None)
    env['PYTHONPATH']=str(site)+os.pathsep+sdk+os.pathsep+env.get('PYTHONPATH','')
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        JULIA_DEPOT_PATH=str(output/'julia-depot')+os.pathsep+str(vendor/'julia-depot'),
        PYTHON_JULIAPKG_EXE=str(vendor/'julia-1.13.3/bin/julia'),
        PYTHON_JULIAPKG_PROJECT=str(project),PYTHON_JULIAPKG_OFFLINE='yes',
        PYTHON_JULIACALL_EXE=str(vendor/'julia-1.13.3/bin/julia'),
        PYTHON_JULIACALL_PROJECT=str(project),JULIA_NUM_THREADS='4',JULIA_PKG_OFFLINE='true',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',JULIA_NUM_PRECOMPILE_TASKS='4')
    # Explicit package bootstrap supports relocated agent trees, without importing the source checkout.
    bootstrap = ("import sys,importlib.util;sys.path.insert(0,"+repr(sdk)+");"
        "p="+repr(str(agent_dir))+";s=importlib.util.spec_from_file_location('agent',p+'/__init__.py',submodule_search_locations=[p]);"
        "m=importlib.util.module_from_spec(s);sys.modules['agent']=m;s.loader.exec_module(m);"
        "from agent.pipeline_bridge.bootstrap import repository;repository();"
        "from agent.pipeline_bridge.backend import install;install(None,sys.argv[1]);"
        "from agent.repository.shore_agent.symbolic_service import _worker;_worker(sys.argv[1])")
    # Use the organizer's working interpreter, not a non-relocatable venv executable.
    python = sys.executable
    pool = WorkerPool(size,python,output,warmup_timeout_seconds=timeout,worker_env=env,
        worker_command_factory=lambda directory:[python,'-B','-u','-c',bootstrap,str(directory)])
    server = None
    try:
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            health=pool.health()
            if health['ready']==size:break
            if health['failed']:raise RuntimeError('PySR startup failed; inspect '+str(output))
            time.sleep(.1)
        else:raise RuntimeError('PySR startup timed out')
        server=make_server(pool,0)
        threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.1},daemon=True).start()
        yield 'http://127.0.0.1:'+str(server.server_port)
    finally:
        if server:
            server.shutdown();server.server_close()
        pool.close()
