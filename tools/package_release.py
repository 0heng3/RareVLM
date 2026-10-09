"""Refresh a release manifest and optionally ZIP source-controlled files."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from datetime import date

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--version',default='1.1')
    p.add_argument('--date',default=date.today().isoformat())
    args=p.parse_args()
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    names=sorted({n for n in names if n and n!='PACKAGE_MANIFEST.json' and (ROOT/n).is_file()})
    if any(Path(n).suffix in ('.pt','.pth','.npz','.bin','.safetensors','.zip','.log') for n in names):raise ValueError('large experiment artifacts may not enter the compact release')
    if any(any(x in Path(n).parts for x in ('.git','.tmp','.cache','data','artifacts','outputs')) for n in names):raise ValueError('generated/private paths in release')
    manifest={'edition':'RareVLM CV and Multimodal Runnable Release','version':args.version,'date':args.date,
              'repository':'https://github.com/0heng3/RareVLM','legacy_clip_study_frozen':True,
              'datasets_and_weights_included':False,
              'files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}
    (ROOT/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for name in names+['PACKAGE_MANIFEST.json']:
                day=date.fromisoformat(args.date)
                info=zipfile.ZipInfo('RareVLM/'+name,date_time=(day.year,day.month,day.day,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o100644<<16
                archive.writestr(info,(ROOT/name).read_bytes())
        digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
        args.output.with_suffix('.zip.sha256').write_text(digest+'  '+args.output.name+'\n')
        print(f'packaged {len(names)+1} files; SHA256 {digest}')
    else:print(f'updated {len(names)} release file hashes')


if __name__=='__main__':main()
