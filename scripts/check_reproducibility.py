"""Repeat numerical experiment and compare evidence CSVs byte-for-byte."""
import os,subprocess,tempfile,hashlib,json,sys
from pathlib import Path
import yaml
root=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((root/'config.yaml').read_text())
with tempfile.TemporaryDirectory(prefix='fingerprint-check-') as tmp:
 cfg['raw_dir']=str(root/'data/raw');cfg['output_dir']=tmp
 config=Path(tmp)/'check.yaml';config.write_text(yaml.safe_dump(cfg))
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 proc=subprocess.run([sys.executable,str(root/'src/pipeline.py'),'--config',str(config)],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 if proc.returncode:raise RuntimeError(proc.stdout)
 files=sorted((root/'results').glob('*.csv'));checks=[]
 for f in files:
  repeat=Path(tmp)/f.name
  ok=repeat.exists() and f.read_bytes()==repeat.read_bytes();checks.append({'file':f.name,'identical':ok})
  if not ok:raise AssertionError('Reproducibility mismatch: '+f.name)
 print(json.dumps({'checked_csvs':len(checks),'all_identical':True,'checks':checks},indent=2))
