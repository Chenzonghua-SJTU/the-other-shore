"""Local original-Docker execution broker; token remains in process memory."""
import base64,hmac,json,os,tempfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from .pipeline_bridge.bootstrap import repository
repository()
from .repository.shore_agent import code_execution as code
TOKEN=os.environ['SHORE_CODE_SERVICE_TOKEN']
CONFIG={'docker_context':'colima-shore-code','image':'shore-code:e2-r04','timeout_seconds':90,'memory_mb':1024}
INFO=code.sandbox_info(CONFIG)
assert INFO['available'],'Original sandbox image unavailable'
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,status,data):
        raw=json.dumps(data,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+TOKEN)
    def do_GET(self):
        if not self.authorized():return self.reply(403,{'error':'unauthorized'})
        if self.path!='/health':return self.reply(404,{})
        self.reply(200,{'version':'original-docker-rpc-v1','sandbox':INFO,'architecture':'arm64'})
    def do_POST(self):
        if not self.authorized():return self.reply(403,{'error':'unauthorized'})
        if self.path!='/execute':return self.reply(404,{})
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=code.MAX_INPUT:raise ValueError('input limit')
            req=json.loads(self.rfile.read(n));assert isinstance(req['code'],str)
            code.validate_output_schema(req['schema'])
            timeout=float(req['timeout_seconds']);assert 1<=timeout<=90
            with tempfile.TemporaryDirectory(prefix='shore-code-broker-') as root:
                out=Path(root)/'artifacts'
                result=code.execute_code(req['code'],req['inputs'],req['schema'],out,CONFIG,timeout,files=req['files'])
                artifacts=[{'name':a['name'],'base64':base64.b64encode((out/a['name']).read_bytes()).decode()} for a in result['artifacts']]
                self.reply(200,{'record':result,'artifacts':artifacts})
        except (ValueError,KeyError,TypeError,AssertionError):self.reply(400,{'error':'invalid_request'})
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',18993),Handler).serve_forever()
