"""Authenticated, idempotent original-Docker jobs; no Docker API exposure."""
import base64,hmac,json,os,tempfile,threading,hashlib,re
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from .pipeline_bridge.bootstrap import repository
repository()
from .repository.shore_agent import code_execution as code
TOKEN=os.environ['SHORE_CODE_SERVICE_TOKEN']
CONFIG={'docker_context':'colima-shore-code','image':'shore-code:e2-r04','timeout_seconds':90,'memory_mb':1024}
INFO=code.sandbox_info(CONFIG);assert INFO['available']
JOBS=OrderedDict();LOCK=threading.Lock();POOL=ThreadPoolExecutor(max_workers=4)
def execute(key,req):
    try:
        with tempfile.TemporaryDirectory(prefix='shore-code-broker-') as root:
            out=Path(root)/'artifacts'
            result=code.execute_code(req['code'],req['inputs'],req['schema'],out,CONFIG,req['timeout_seconds'],files=req['files'])
            artifacts=[{'name':a['name'],'base64':base64.b64encode((out/a['name']).read_bytes()).decode()} for a in result['artifacts']]
            response={'record':result,'artifacts':artifacts}
        with LOCK:JOBS[key]['response']=response;JOBS[key]['state']='done'
    except Exception:
        with LOCK:JOBS[key]['state']='failed'
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,status,data):
        raw=json.dumps(data,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers()
        try:self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError):pass
    def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+TOKEN)
    def do_GET(self):
        if not self.authorized():return self.reply(403,{'error':'unauthorized'})
        if self.path=='/health':return self.reply(200,{'version':'original-docker-rpc-v1','sandbox':INFO,'architecture':'arm64'})
        if self.path.startswith('/result/'):
            key=self.path.split('/')[-1]
            with LOCK:job=JOBS.get(key);data={'state':job['state'],'response':job.get('response')} if job else None
            if data is None:return self.reply(404,{})
            if data['state']=='failed':return self.reply(503,{'error':'execution_backend_failed'})
            return self.reply(200,data)
        self.reply(404,{})
    def do_POST(self):
        if not self.authorized():return self.reply(403,{'error':'unauthorized'})
        if self.path!='/execute':return self.reply(404,{})
        try:
            n=int(self.headers.get('Content-Length','0'));assert 0<n<=code.MAX_INPUT
            req=json.loads(self.rfile.read(n));assert isinstance(req['code'],str)
            key=req['job_id'];assert re.fullmatch('[0-9a-f]{32}',key)
            code.validate_output_schema(req['schema']);assert 1<=float(req['timeout_seconds'])<=90
            digest=hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()
            with LOCK:
                if key in JOBS:assert JOBS[key]['digest']==digest
                else:
                    while len(JOBS)>=32:
                        old=next((k for k,v in JOBS.items() if v['state']!='pending'),None)
                        if old is None:raise ValueError('queue_full')
                        del JOBS[old]
                    JOBS[key]={'state':'pending','digest':digest};POOL.submit(execute,key,req)
            self.reply(200,{'job_id':key})
        except (ValueError,KeyError,TypeError,AssertionError):self.reply(400,{'error':'invalid_request'})
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',18993),Handler).serve_forever()
