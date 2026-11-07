"""Transport contracts: authentication, once-only jobs, immutable request IDs."""
import json,os,threading,unittest,urllib.request,urllib.error,importlib
from unittest.mock import patch,MagicMock
from http.server import ThreadingHTTPServer

class BrokerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .repository.shore_agent import code_execution
        with patch.dict(os.environ,{'SHORE_CODE_SERVICE_TOKEN':'unit-test-only'}),patch.object(code_execution,'sandbox_info',return_value={'available':True,'image_id':'sha256:'+'0'*64}):
            cls.b=importlib.import_module('agent.code_broker')
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),cls.b.Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url='http://127.0.0.1:'+str(cls.server.server_port)
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.b.POOL.shutdown()
    def setUp(self):self.b.JOBS.clear()
    def request(self,path,data=None,auth=True):
        headers={'Authorization':'Bearer unit-test-only'} if auth else {}
        req=urllib.request.Request(self.url+path,data=None if data is None else json.dumps(data).encode(),headers=headers)
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=2) as r:return json.load(r)
    def payload(self):return {'job_id':'a'*32,'code':'result=1','schema':{},'inputs':{},'files':{},'timeout_seconds':10}
    def test_authentication_required(self):
        with self.assertRaises(urllib.error.HTTPError) as error:self.request('/health',auth=False)
        self.assertEqual(error.exception.code,403)
    def test_retries_submit_exactly_once(self):
        with patch.object(self.b.POOL,'submit') as submit:
            self.request('/execute',self.payload());self.request('/execute',self.payload())
            self.assertEqual(submit.call_count,1)
            self.assertEqual(self.request('/result/'+'a'*32)['state'],'pending')
    def test_request_id_cannot_change_code(self):
        with patch.object(self.b.POOL,'submit') as submit:
            self.request('/execute',self.payload());changed=self.payload();changed['code']='result=2'
            with self.assertRaises(urllib.error.HTTPError) as error:self.request('/execute',changed)
            self.assertEqual(error.exception.code,400);self.assertEqual(submit.call_count,1)
    def test_completed_response_is_reused(self):
        with patch.object(self.b.POOL,'submit') as submit:
            self.request('/execute',self.payload());self.b.JOBS['a'*32].update(state='done',response={'record':{'result':42}})
            self.request('/execute',self.payload())
            self.assertEqual(self.request('/result/'+'a'*32)['response']['record']['result'],42)
            self.assertEqual(submit.call_count,1)
if __name__=='__main__':unittest.main()
