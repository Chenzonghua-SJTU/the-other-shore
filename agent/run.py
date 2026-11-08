"""Three environment-interaction batches with explicit model-authored laws."""
import argparse,copy,json,os,signal,sys
sys.dont_write_bytecode=True
from pathlib import Path
from rift import Console
from rift.strategy import load_model_registry,model_binding
from .models import managed_model
from .pipeline_bridge.bootstrap import repository
repository()
from .pipeline_bridge.interaction import InteractionAgent as Agent
from .repository.shore_agent.runtime import ToolRuntime
from .pipeline_bridge.model_transport import DeepSeekClient
from .pipeline_bridge.environment import Environment
from .pipeline_bridge.backend import install
from .pipeline_bridge.service import symbolic_service
from .pipeline_bridge.reliability import configure,ProtocolClient
from .pipeline_bridge.lifecycle import SubmissionLifecycle

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models-file',default=os.environ.get('SHORE_MODEL_REGISTRY'))
    p.add_argument('--model-ref');p.add_argument('--base-url');p.add_argument('--model-id')
    p.add_argument('--model-python');p.add_argument('--startup-timeout',type=float,default=900.)
    p.add_argument('--output-dir',default=os.environ.get('RIFT_OUTPUT_DIR'))
    p.add_argument('--symbolic-endpoint');p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--max-turns',type=int,default=3,
        help='Environment-interaction batches (1..3); local fitting/reasoning does not spend a batch')
    p.add_argument('--tp-size',type=int,default=4)
    p.add_argument('--code-endpoint')
    args=p.parse_args(argv)
    if not args.output_dir:p.error('RIFT_OUTPUT_DIR required')
    if not 1<=args.max_tokens<=4096:p.error('max-tokens must be 1..4096')
    if not 1<=args.max_turns<=3:p.error('max-turns must be 1..3; unlimited or extra interaction batches are not allowed')
    A=Path(__file__).resolve().parent;out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    from .pipeline_bridge.admission import acquire as acquire_resource
    resource_lease=acquire_resource(A,out,args)
    os.environ['RIFT_OUTPUT_DIR']=str(out)
    (out/'tmp').mkdir(exist_ok=True);os.environ['TMPDIR']=str(out/'tmp')
    config=json.loads((A/'repository-config.json').read_text())
    config['agent']['max_turns']=args.max_turns
    config['agent']['counting_policy']='environment_batches'
    # At most one transport retry; all attempts are logged and spend wall time.
    config['model']['max_retries']=1
    if config['code_execution']['enabled'] and args.code_endpoint:
        from .pipeline_bridge.code_transport import install as install_code_transport
        receipt=install_code_transport(args.code_endpoint)
        (out/'code-backend.json').write_text(json.dumps(receipt,indent=2))
    binding=model_binding(load_model_registry(args.models_file or A/'models.json',check_files=True),args.model_ref)
    lifecycle=SubmissionLifecycle()
    previous=signal.signal(signal.SIGTERM,lifecycle.handle_sigterm)
    try:
        with managed_model(binding['base_model'],base_url=args.base_url,model_id=args.model_id,
            model_path=binding['path'],adapter_path=binding.get('adapter_path'),python_executable=args.model_python,
            tensor_parallel_size=args.tp_size,max_model_len=262144,startup_timeout=args.startup_timeout,log_dir=out/'model-logs') as model:
            with symbolic_service(out,timeout=args.startup_timeout,external_endpoint=args.symbolic_endpoint) as endpoint:
                env=lifecycle.observe(Environment(Console()));env.ruleset=config['environment']['ruleset']
                backend=install(env,out)
                numerics=copy.deepcopy(config['numerics']);numerics['symbolic_regression_endpoint']=endpoint
                runtime=ToolRuntime(env,out,config=numerics,code_config=config['code_execution'])
                backend.attach(runtime)
                configure(runtime)
                import hashlib,inspect
                provenance=json.loads((A/'repository-source.json').read_text())
                (out/'runtime-provenance.json').write_text(json.dumps({'github_commit':provenance['github_commit'],
                    'agent_class':Agent.__module__,'agent_file':inspect.getfile(Agent),
                    'runtime_class':ToolRuntime.__module__,'runtime_file':inspect.getfile(ToolRuntime),
                    'client_class':DeepSeekClient.__module__,'fit_file':inspect.getfile(runtime.fit_law_and_rollout),
                    'tool_schemas_sha256':hashlib.sha256(json.dumps(runtime.schemas,sort_keys=True).encode()).hexdigest(),
                    'protocol_overlay':'interaction-v17: joint background parameter validation, calibrated model timeout, schema preflight, early candidate registration, bounded repeats, deadline reserve, complete named-tool SSE compatibility; vendored source unchanged',
                    'max_turns':args.max_turns,'counting_policy':'environment_batches',
                    'trajectory_backend':'official rift.rollout','scientific_source_files':len(provenance['files'])},indent=2))

                settings=copy.deepcopy(config['model']);settings.update(base_url=model.base_url,model=model.model,
                    max_tokens_per_call=args.max_tokens,local_model_endpoints={model.model:model.base_url})
                (out/'effective_config.json').write_text(json.dumps({**config,'model':settings,'numerics':numerics},ensure_ascii=False,indent=2))
                client=DeepSeekClient(settings,out,remaining_seconds=lambda:runtime.budget()['wall_seconds_remaining'])
                try:result=Agent(ProtocolClient(client),runtime,out,config['agent'],
                    max_turns=args.max_turns,counting_policy='environment_batches').run()
                finally:client.close()
                (out/'bridge-summary.json').write_text(json.dumps({'official_rollouts':backend.rollout_calls,'repository_integrity':True}))
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return 0 if lifecycle.confirmed and result.get('status')=='submitted' else 2
    finally:
        lifecycle.write_receipt(out/'exit-lifecycle.json')
        signal.signal(signal.SIGTERM,previous)
if __name__=='__main__':raise SystemExit(main())
