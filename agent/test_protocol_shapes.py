"""Canonical repository wire schema for every public world, using only official observations."""
import json,os,tempfile
from pathlib import Path
from rift import Console
from development_worlds.console import DevelopmentConsole
from development_worlds.service import DevelopmentSession,serve_session
from development_worlds import catalog
from .pipeline_bridge.bootstrap import repository
repository()
from .pipeline_bridge.environment import Environment
from .repository.shore_agent.runtime import ToolRuntime

def main():
    A=Path(__file__).parent;out=Path(tempfile.mkdtemp(prefix='protocol-shapes-',dir=A/'_tests')).resolve();os.environ['RIFT_OUTPUT_DIR']=str(out)
    expected=json.loads((A/'_tests/r05-protocol-reference.json').read_text());results=[]
    for world in map(str,range(1,7)):
        session=DevelopmentSession(DevelopmentConsole(world,catalog.noise_sigma(world),42,catalog.config(world)),out/world/'session')
        with serve_session(session) as credentials:
            env=Environment(Console(credentials['endpoint'],credentials['token']));runtime=ToolRuntime(env,out/world/'runtime',config={'symbolic_regression_enabled':False})
            response=runtime.mission();assert response['ok'],response;data=response['data']
            assert set(data['task'])==set(expected['task_keys'])
            for probe in data['first_light']['probes']:
                assert set(probe)==set(expected['probe_keys'])
                assert len(probe['times'])==len(probe['r_obs'])==len(probe['v_obs'])==8
                assert set(probe['initial_conditions'])=={'r0','v0','properties','t0'}
                assert set(probe['initial_conditions']['properties'])==set(data['task']['allowed_properties'])
            assert set(data['first_light']['body_observation'])==set(expected['body_keys'])
            assert abs(data['task']['position_noise_sigma']-catalog.noise_sigma(world))<1e-12
            obs=runtime.execute('observe_body',{'times':[0.,.1]});assert obs['ok'],obs
            assert '_official_response' not in obs['data']
            results.append({'world':world,'task_schema_matches_repository':True,'full_first_light':True,'public_sigma_matches':True})
    repository();(out/'report.json').write_text(json.dumps(results,indent=2));print('PASS',out,json.dumps(results))
if __name__=='__main__':main()
