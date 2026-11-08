"""Loopback RPC transport to the original Docker sandbox; no host-code fallback."""
import base64,json,os,time,urllib.request,urllib.error,uuid
from pathlib import Path
from urllib.parse import urlparse
from ..repository.shore_agent import code_execution as original
VERSION='original-docker-rpc-v1'

def install(endpoint):
    endpoints=list(dict.fromkeys([endpoint]+[e for e in os.environ.get('SHORE_CODE_SERVICE_ENDPOINTS','').split(',') if e]))
    for item in endpoints:
        u=urlparse(item)
        if u.scheme!='http' or u.hostname!='127.0.0.1' or u.username or u.password or u.path not in ('','/'):
            raise ValueError('Code broker must be an SSH-forwarded loopback endpoint')
    preferred=0;failovers=0
    token=os.environ.get('SHORE_CODE_SERVICE_TOKEN')
    if not token:raise ValueError('Code broker authentication unavailable')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path,payload=None,timeout=20):
        nonlocal preferred,failovers
        raw=None if payload is None else json.dumps(payload,allow_nan=False).encode()
        deadline=time.monotonic()+timeout
        while True:
            req=urllib.request.Request(endpoints[preferred].rstrip('/')+path,data=raw,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            try:
                with opener.open(req,timeout=min(3,max(.1,deadline-time.monotonic()))) as r:
                    data=r.read(original.MAX_RESPONSE+1)
                if len(data)>original.MAX_RESPONSE:raise ValueError('Code response too large')
                return json.loads(data)
            except urllib.error.HTTPError:
                raise
            except (OSError,urllib.error.URLError):
                if time.monotonic()>=deadline:raise
                preferred=(preferred+1)%len(endpoints);failovers+=1
                time.sleep(.1)
    health=request('/health')
    if health.get('version')!=VERSION or not health.get('sandbox',{}).get('available'):
        raise ValueError('Original Docker broker unavailable')
    expected_image=health['sandbox']['image_id']
    def sandbox_info(config=None):return dict(health['sandbox'],transport=VERSION)
    def execute_code(code,inputs,schema,artifact_dir,config,timeout_seconds,files=None):
        started=time.monotonic();initial_failovers=failovers;allowed=min(float(config.get('timeout_seconds',90)),float(timeout_seconds))
        if allowed<1:raise ValueError('code_execution_time_exhausted')
        job_id=uuid.uuid4().hex
        deadline=started+allowed+35
        payload={'job_id':job_id,'code':code,'inputs':inputs,'schema':schema,'files':files or {},'timeout_seconds':allowed}
        request('/execute',payload,timeout=min(20,max(.1,deadline-time.monotonic())))
        while True:
            state=request('/result/'+job_id,timeout=min(20,max(.1,deadline-time.monotonic())))
            if state['state']=='done':
                raw=state['response'];break
            if time.monotonic()>=deadline:raise ValueError('Code RPC wall-time deadline exceeded')
            time.sleep(.25)
        rec=raw['record'];status=rec['status']
        if status not in ('success','error','schema_error','timeout','memory_limit','output_limit','protocol_error'):raise ValueError('Invalid code status')
        if rec['sandbox']['image_id']!=expected_image or rec['sandbox']['network']!='none' or rec['sandbox']['host_mounts'] is not False:raise ValueError('Sandbox receipt mismatch')
        # Reuse the upstream atomic artifact/output validator on the receiver.
        wire={k:rec.get(k) for k in ('result','error','stdout','stderr','warnings')}
        wire['warnings']=rec.get('warnings',[])
        wire.update(status='success' if status=='success' else 'error',artifacts=raw['artifacts'])
        for k in ('compute_seconds','cpu_seconds_reported','peak_rss_kib_reported'):wire[k]=rec.get(k,0)
        result=original._decode(wire,Path(artifact_dir),schema)
        result.update(status=status if result['status']!='schema_error' else 'schema_error',exit_code=rec.get('exit_code'),
                      elapsed_seconds=time.monotonic()-started,remote_elapsed_seconds=rec['elapsed_seconds'],sandbox=dict(rec['sandbox'],transport=VERSION,transport_endpoint=endpoints[preferred],transport_failovers=failovers-initial_failovers))
        if rec.get('cleanup_diagnostic'):result['cleanup_diagnostic']=rec['cleanup_diagnostic']
        return result
    original.sandbox_info=sandbox_info
    original.execute_code=execute_code
    return {'version':VERSION,'endpoints':endpoints,'health_endpoint':endpoints[preferred],'startup_failovers':failovers,'sandbox':health['sandbox'],'docker_architecture':health['architecture'],'broker_implementation_sha256':health.get('implementation_sha256'),'broker_cleanup_sha256':health.get('cleanup_sha256')}
