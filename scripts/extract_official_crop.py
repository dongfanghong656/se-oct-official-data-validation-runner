#!/usr/bin/env python3
"""Download the public phase-corrected TiO2 volume and save reusable validation crops.

The deposited Cscan follows the authors' MATLAB convention [z, x, y], not [z, y, x].
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np
import requests
from scipy.io import loadmat, savemat

URL='https://zenodo.org/records/7870795/files/input_TiO2gelatin_004_phasecorrected.mat?download=1'
EXPECTED_MD5='d905784f950a6fadfce71e9efc6d2654'
# Legacy exploratory crop retained for provenance; axis names corrected below.
LEGACY_X0=232; LEGACY_Y0=210; LEGACY_HALF=48
# White-arrow point used for the out-of-focus example in plot_figure3.m:
# x=0.1985 mm, y=0.0975 mm over a 0.225-mm/512-sample field.
FIG4_X0=451; FIG4_Y0=221  # zero-based nearest indices
FIG4_X_START=331; FIG4_X_STOP=512  # 181 samples, target local x=120
FIG4_Y_START=131; FIG4_Y_STOP=312  # 181 samples, target local y=90

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

def write_crop(out:Path,name:str,crop:np.ndarray,sk:np.ndarray,metadata:dict)->dict:
    path=out/name
    payload={'Ccrop':np.asarray(crop,dtype=np.complex64),'sk':sk}
    for key,value in metadata.items():
        if isinstance(value,int): payload[key]=np.int32(value)
    savemat(path,payload,do_compression=True)
    return {'crop_file':path.name,'crop_sha256':digest(path,'sha256'),'crop_bytes':path.stat().st_size,'crop_shape':list(crop.shape),**metadata}

def main()->None:
    root=Path(__file__).resolve().parents[1]
    cache=root/'.cache'/'input_TiO2gelatin_004_phasecorrected.mat'
    out=root/'reports'/'extracts'; out.mkdir(parents=True,exist_ok=True); cache.parent.mkdir(exist_ok=True)
    download(cache)
    mat=loadmat(cache,variable_names=['Cscan','sk'])
    C=np.asarray(mat['Cscan'],dtype=np.complex64); sk=np.asarray(mat['sk']).reshape(-1)
    if C.shape!=(200,512,512):raise RuntimeError(f'unexpected shape {C.shape}')

    # C axes are [z, x, y], matching [Nz,Nx,Ny] in MIAA_ISAM_processing.m.
    legacy=C[:,LEGACY_X0-LEGACY_HALF:LEGACY_X0+LEGACY_HALF+1,LEGACY_Y0-LEGACY_HALF:LEGACY_Y0+LEGACY_HALF+1]
    legacy_info=write_crop(out,'official_tio2_crop_97.mat',legacy,sk,{
      'x0':LEGACY_X0,'y0':LEGACY_Y0,'half':LEGACY_HALF,'axis_order':'z,x,y','status':'legacy_exploratory_not_figure4_target'})

    fig4=C[:,FIG4_X_START:FIG4_X_STOP,FIG4_Y_START:FIG4_Y_STOP]
    fig4_info=write_crop(out,'official_tio2_fig4_outfocus_crop.mat',fig4,sk,{
      'x0':FIG4_X0,'y0':FIG4_Y0,'x_start':FIG4_X_START,'x_stop':FIG4_X_STOP,
      'y_start':FIG4_Y_START,'y_stop':FIG4_Y_STOP,'target_local_x':FIG4_X0-FIG4_X_START,
      'target_local_y':FIG4_Y0-FIG4_Y_START,'expected_final_z_index_matlab1':461,
      'axis_order':'z,x,y','source_coordinate_mm':'x=0.1985,y=0.0975 from plot_figure3.m'})

    manifest={'source_record':'10.5281/zenodo.7870795','source_file':cache.name,'source_md5':digest(cache,'md5'),
      'source_boundary':'phase-corrected complex OCT C-scan, not raw camera interferograms',
      'cscan_axis_order':'z,x,y','crops':[legacy_info,fig4_info]}
    (out/'official_tio2_crops_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Backward-compatible legacy manifest, now explicit about axis semantics.
    (out/'official_tio2_crop_97_manifest.json').write_text(json.dumps({'source_record':manifest['source_record'],'source_file':cache.name,'source_md5':manifest['source_md5'],'source_boundary':manifest['source_boundary'],**legacy_info},indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
