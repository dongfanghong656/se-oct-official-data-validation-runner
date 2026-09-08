#!/usr/bin/env python3
"""Download the public phase-corrected TiO2 volume and save a reusable 97x97 crop."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np
import requests
from scipy.io import loadmat, savemat

URL='https://zenodo.org/records/7870795/files/input_TiO2gelatin_004_phasecorrected.mat?download=1'
EXPECTED_MD5='d905784f950a6fadfce71e9efc6d2654'
Y0=232; X0=210; HALF=48

def digest(path:Path,alg:str)->str:
    h=hashlib.new(alg)
    with path.open('rb') as f:
        while True:
            c=f.read(4*1024*1024)
            if not c: break
            h.update(c)
    return h.hexdigest()

def download(path:Path)->None:
    if path.exists() and digest(path,'md5')==EXPECTED_MD5:return
    part=path.with_suffix(path.suffix+'.part')
    for attempt in range(1,8):
        try:
            off=part.stat().st_size if part.exists() else 0
            headers={'Range':f'bytes={off}-'} if off else {}
            mode='ab' if off else 'wb'
            with requests.get(URL,stream=True,timeout=(30,240),headers=headers) as r:
                if r.status_code==200 and off: mode='wb'
                r.raise_for_status()
                with part.open(mode) as f:
                    for c in r.iter_content(4*1024*1024):
                        if c:f.write(c)
            part.replace(path)
            actual=digest(path,'md5')
            if actual!=EXPECTED_MD5: raise RuntimeError(f'MD5 mismatch {actual}')
            return
        except Exception:
            if attempt==7:raise
            time.sleep(min(90,2**attempt))

def main()->None:
    root=Path(__file__).resolve().parents[1]
    cache=root/'.cache'/'input_TiO2gelatin_004_phasecorrected.mat'
    out=root/'reports'/'extracts'; out.mkdir(parents=True,exist_ok=True); cache.parent.mkdir(exist_ok=True)
    download(cache)
    mat=loadmat(cache,variable_names=['Cscan','sk'])
    C=np.asarray(mat['Cscan'],dtype=np.complex64); sk=np.asarray(mat['sk']).reshape(-1)
    if C.shape!=(200,512,512):raise RuntimeError(f'unexpected shape {C.shape}')
    crop=C[:,Y0-HALF:Y0+HALF+1,X0-HALF:X0+HALF+1]
    path=out/'official_tio2_crop_97.mat'
    savemat(path,{'Ccrop':crop,'sk':sk,'y0':np.int32(Y0),'x0':np.int32(X0),'half':np.int32(HALF)},do_compression=True)
    manifest={'source_record':'10.5281/zenodo.7870795','source_file':cache.name,'source_md5':digest(cache,'md5'),
      'source_boundary':'phase-corrected complex OCT C-scan, not raw camera interferograms',
      'crop_file':path.name,'crop_sha256':digest(path,'sha256'),'crop_bytes':path.stat().st_size,
      'crop_shape':list(crop.shape),'center_yx0':[Y0,X0],'half_width':HALF}
    (out/'official_tio2_crop_97_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
