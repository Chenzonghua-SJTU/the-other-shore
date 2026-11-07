"""Load the immutable original SDK extracted inside /agent; outputs stay in /agent."""
import pathlib,json,zipfile,hashlib,sys,importlib.util,os,time
A=pathlib.Path(__file__).resolve().parent
baseline=A/'_experiments/E13-20261108-001604/report.json'
print('VERIFY_PREVIOUS_AUTOMATIC_PILOT',str(baseline),flush=True)
while not baseline.exists():time.sleep(2)
registered=json.loads(baseline.read_text())
assert registered.get('baseline_complete') and registered.get('source_integrity') and registered.get('official_files_unchanged'), 'Baseline incomplete or integrity failed; do not run ablation'
print('BASELINE_VERIFIED',len(registered['summary_rows']),flush=True)
C=A/'_runtime/official-pristine-BF2';C.mkdir(parents=True,exist_ok=True)
z=zipfile.ZipFile(A.parent.parent/'shore-contest-starter-v15.zip');expected={}
for name in z.namelist():
 if name.endswith('/'):continue
 parts=pathlib.PurePosixPath(name).parts
 if parts[0]=='shore-contest-starter-v15':parts=parts[1:]
 if not parts or parts[0]=='agent':continue
 assert all(p not in ('..','.') for p in parts)
 target=C.joinpath(*parts);assert target.is_relative_to(C)
 data=z.read(name);expected[str(pathlib.Path(*parts))]=hashlib.sha256(data).hexdigest()
 if target.exists():assert target.read_bytes()==data,str(target)
 else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
assert len(expected)==84
manifest=json.loads((A/'repository-source.json').read_text())
assert all(hashlib.sha256((A/'repository/shore_agent'/k).read_bytes()).hexdigest()==v for k,v in manifest['files'].items())
files=['run.py','pipeline_bridge/backend.py','pipeline_bridge/compiler.py','pipeline_bridge/reliability.py','repository-config.json','pipeline_bridge/trajectory_filter.py','pipeline_bridge/filter_tool.py','pipeline_bridge/filter_capability.py','pipeline_bridge/service.py','pipeline_bridge/filter_options.py','pipeline_bridge/filter_guidance.py','pipeline_bridge/automatic_filter.py']
meta={'version':'BF2+run_code-original-docker-rpc','base_commit':manifest['github_commit'],'original_scientific_package_unchanged':True,'runtime_overlay':True,'files':{k:hashlib.sha256((A/k).read_bytes()).hexdigest() for k in files},'official_root':str(C),'official_zip_files':expected,'fixed_files_match_original_zip':True,'sampling_interval':'advisory only; no changed request or official integrator','changes':['remove impossible submission gate','isolated property and velocity-noise diagnostics','one-time evidence review','unambiguous batch-t0 correction','autonomous shared-RHS background hypothesis','noise-aware sampling and long-span guidance']}
(A/'_bugfixes/CODE_ABLATION1').mkdir(parents=True,exist_ok=True)
(A/'_bugfixes/CODE_ABLATION1/manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
os.environ['SHORE_SDK_ROOT']=str(C);os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(C));s=importlib.util.spec_from_file_location('agent',A/'__init__.py',submodule_search_locations=[str(A)]);m=importlib.util.module_from_spec(s);sys.modules['agent']=m;s.loader.exec_module(m)
import agent.benchmark_official as bo
import agent.benchmark_code3 as b
bo.ROOT=C;b.ROOT=C
sys.argv=['benchmark_e15','--round-start','1','--bugfix','BF2-run-code-v1']
b.main()
