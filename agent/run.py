"""Repository Agent unchanged; official environment/rollout/export bridge only."""
import argparse,copy,json,os,signal,sys
sys.dont_write_bytecode=True
from pathlib import Path
from rift import Console
from rift.strategy import load_model_registry,model_binding
from .models import managed_model
from .pipeline_bridge.bootstrap import repository
repository()
from .repository.shore_agent.agent import Agent
from .repository.shore_agent.runtime import ToolRuntime
from .repository.shore_agent.model import DeepSeekClient
from .pipeline_bridge.environment import Environment
from .pipeline_bridge.backend import install
from .pipeline_bridge.service import symbolic_service
from .pipeline_bridge.reliability import configure,ProtocolClient
from .pipeline_bridge.filter_options import add_filter_options,resolve_filter_enabled,resolve_filter_policy

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models-file',default=os.environ.get('SHORE_MODEL_REGISTRY'))
    p.add_argument('--model-ref');p.add_argument('--base-url');p.add_argument('--model-id')
    p.add_argument('--model-python');p.add_argument('--startup-timeout',type=float,default=900.)
    p.add_argument('--output-dir',default=os.environ.get('RIFT_OUTPUT_DIR'))
    p.add_argument('--symbolic-endpoint');p.add_argument('--max-tokens',type=int,default=4096)
    add_filter_options(p)
    code_flags=p.add_mutually_exclusive_group()
    code_flags.add_argument('--run-code',dest='run_code',action='store_true')
    code_flags.add_argument('--no-run-code',dest='run_code',action='store_false')
    p.set_defaults(run_code=None)
    p.add_argument('--code-endpoint')
    p.add_argument('--max-turns',type=int,default=None,help='Legacy launcher parameter, unused by repository Agent')
    p.add_argument('--tp-size',type=int,default=4)
    args=p.parse_args(argv)
    if not args.output_dir:p.error('RIFT_OUTPUT_DIR required')
    if not 1<=args.max_tokens<=4096:p.error('max-tokens must be 1..4096')
    A=Path(__file__).resolve().parent;out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    from .pipeline_bridge.admission import acquire as acquire_resource
    resource_lease=acquire_resource(A,out,args)
    os.environ['RIFT_OUTPUT_DIR']=str(out)
    (out/'tmp').mkdir(exist_ok=True);os.environ['TMPDIR']=str(out/'tmp')
    config=json.loads((A/'repository-config.json').read_text())
    if args.run_code is not None:config['code_execution']['enabled']=args.run_code
    code_receipt=None
    if config['code_execution']['enabled'] and args.code_endpoint:
        from .pipeline_bridge.code_transport import install as install_code_transport
        code_receipt=install_code_transport(args.code_endpoint)
    filter_cli_override=args.trajectory_filter
    filter_policy_cli_override=args.trajectory_filter_policy
    try:
        args.trajectory_filter=resolve_filter_enabled(config['numerics'],filter_cli_override)
        args.trajectory_filter_policy=resolve_filter_policy(config['numerics'],filter_policy_cli_override)
    except ValueError as exc:p.error(str(exc))
    binding=model_binding(load_model_registry(args.models_file or A/'models.json',check_files=True),args.model_ref)
    def stop(signum,frame):raise SystemExit(128+signum)
    previous=signal.signal(signal.SIGTERM,stop)
    try:
        with managed_model(binding['base_model'],base_url=args.base_url,model_id=args.model_id,
            model_path=binding['path'],adapter_path=binding.get('adapter_path'),python_executable=args.model_python,
            tensor_parallel_size=args.tp_size,max_model_len=262144,startup_timeout=args.startup_timeout,log_dir=out/'model-logs') as model:
            service_options={'trajectory_filter':True} if args.trajectory_filter else {}
            with symbolic_service(out,timeout=args.startup_timeout,external_endpoint=args.symbolic_endpoint,**service_options) as endpoint:
                env=Environment(Console());env.ruleset=config['environment']['ruleset']
                backend=install(env,out)
                numerics=copy.deepcopy(config['numerics']);numerics['symbolic_regression_endpoint']=endpoint
                if 'trajectory_filter_enabled' in numerics or filter_cli_override is not None:
                    numerics['trajectory_filter_enabled']=args.trajectory_filter
                if 'trajectory_filter_policy' in numerics or filter_policy_cli_override is not None:
                    numerics['trajectory_filter_policy']=args.trajectory_filter_policy
                runtime=ToolRuntime(env,out,config=numerics,code_config=config['code_execution'])
                backend.attach(runtime)
                configure(runtime)
                if args.trajectory_filter:
                    from .pipeline_bridge.filter_tool import install as install_filter
                    install_filter(runtime,enabled=True,policy=args.trajectory_filter_policy)
                    numerics['trajectory_filter_tool']='joint_spline_v1'
                import hashlib,inspect
                provenance=json.loads((A/'repository-source.json').read_text())
                (out/'runtime-provenance.json').write_text(json.dumps({'github_commit':provenance['github_commit'],
                    'agent_class':Agent.__module__,'agent_file':inspect.getfile(Agent),
                    'runtime_class':ToolRuntime.__module__,'runtime_file':inspect.getfile(ToolRuntime),
                    'client_class':DeepSeekClient.__module__,'fit_file':runtime.fit_law_and_rollout.__func__.__code__.co_filename,
                    'tool_schemas_sha256':hashlib.sha256(json.dumps(runtime.schemas,sort_keys=True).encode()).hexdigest(),
                    'protocol_overlay':'BF2: logged interface/research feedback corrections; original source unchanged','trajectory_backend':'official rift.rollout','scientific_source_files':len(provenance['files'])},indent=2))
                if code_receipt:
                    (out/'code-backend-receipt.json').write_text(json.dumps(code_receipt,indent=2))
                if args.trajectory_filter:
                    provenance_path=out/'runtime-provenance.json'
                    provenance_record=json.loads(provenance_path.read_text())
                    provenance_record['trajectory_filter_tool']='joint_spline_v1; opt-in adapter overlay; original scientific source hashes unchanged'
                    provenance_record['trajectory_filter_policy']=args.trajectory_filter_policy
                    if args.trajectory_filter_policy=='explicit':
                        from .pipeline_bridge.filter_guidance import VERSION as filter_guidance_version
                        provenance_record['trajectory_filter_guidance']=filter_guidance_version
                    provenance_path.write_text(json.dumps(provenance_record,indent=2))

                settings=copy.deepcopy(config['model']);settings.update(base_url=model.base_url,model=model.model,
                    max_tokens_per_call=args.max_tokens,local_model_endpoints={model.model:model.base_url})
                (out/'effective_config.json').write_text(json.dumps({**config,'model':settings,'numerics':numerics},ensure_ascii=False,indent=2))
                client=DeepSeekClient(settings,out,remaining_seconds=lambda:runtime.budget()['wall_seconds_remaining'])
                agent_client=ProtocolClient(client)
                if args.trajectory_filter and args.trajectory_filter_policy=='explicit':
                    from .pipeline_bridge.filter_guidance import configure as configure_filter_guidance
                    agent_client=configure_filter_guidance(agent_client,enabled=True)
                try:result=Agent(agent_client,runtime,out,config['agent']).run()
                finally:client.close()
                (out/'bridge-summary.json').write_text(json.dumps({'official_rollouts':backend.rollout_calls,'repository_integrity':True}))
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return 0 if result.get('status')=='submitted' else 2
    finally:signal.signal(signal.SIGTERM,previous)
if __name__=='__main__':raise SystemExit(main())
