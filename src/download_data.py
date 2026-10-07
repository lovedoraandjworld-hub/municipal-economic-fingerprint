"""Restore official archive with fixed SHA256; never accept an unverified mirror."""
import hashlib,json,urllib.request,zipfile,io,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'data/manifest.json').read_text())
errors=[]
for u in [m['source_url'],m['download_url']]:
 try:
  with urllib.request.urlopen(u,timeout=60) as r:data=r.read()
  if hashlib.sha256(data).hexdigest()!=m['archive_sha256']:raise ValueError('Archive SHA256 mismatch; source version changed')
  z=zipfile.ZipFile(io.BytesIO(data));dest=root/'data/raw';dest.mkdir(parents=True,exist_ok=True)
  for filename,sha in m['files'].items():
   candidates=[n for n in z.namelist() if n.split('/')[-1]==filename]
   if len(candidates)!=1:raise ValueError('Missing or ambiguous member '+filename)
   content=z.read(candidates[0])
   if hashlib.sha256(content).hexdigest()!=sha:raise ValueError('File checksum mismatch '+filename)
   (dest/filename).write_bytes(content)
  print('Official source restored, checksums verified.');break
 except Exception as e:errors.append(f'{u}: {e}')
else:raise RuntimeError('\n'.join(errors))
