#!/usr/bin/env python3
"""Extract and independently refit the exact Figure-5 MIAA+ISAM A-line from authors' processed output."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.optimize import least_squares

NAME='exp_leaf_image_MIAA_ISAM.mat'
MD5='239a1e714b77e6aa6640f4dd2f50a644'
URL=f'https://zenodo.org/records/7870795/files/{NAME}?download=1'
X0=162  # Ascan_xind=163 MATLAB 1-based
Y0=255  # Yindex=256 MATLAB 1-based
LMIN=5.011840350737296e-7;LMAX=5.25482346025955e-7;NREF=1.33;NISAM=1029
KMIN=2*np.pi/LMAX*NREF;KMAX=2*np.pi/LMIN*NREF
ZMAX_MM=np.pi*(200-1)/(KMAX-KMIN)*1e3
ZFULL_MM=np.linspace(0,ZMAX_MM,NISAM)
ZSAVED_MM=ZFULL_MM[199:1000]

def digest(p:Path)->str:
 h=hashlib.md5()
 with p.open('rb') as f:
  while c:=f.read(8*1024*1024):h.update(c)
 return h.hexdigest()
def download(p:Path):
 if p.exists() and digest(p)==MD5:return
 part=p.with_suffix('.part');p.parent.mkdir(parents=True,exist_ok=True)
 for a in range(1,8):
  try:
   off=part.stat().st_size if part.exists() else 0;headers={'Range':f'bytes={off}-'} if off else {};mode='ab' if off else 'wb'
   with requests.get(URL,stream=True,timeout=(30,300),headers=headers) as r:
    if r.status_code==200 and off:mode='wb'
    r.raise_for_status()
    with part.open(mode) as f:
     for c in r.iter_content(8*1024*1024):
      if c:f.write(c)
   part.replace(p);actual=digest(p)
   if actual!=MD5:raise RuntimeError(actual)
   return
  except Exception:
   if a==7:raise
   time.sleep(min(120,2**a))
def dataset(h5):
 c=[]
 h5.visititems(lambda n,o:c.append(o) if isinstance(o,h5py.Dataset) and o.ndim==3 else None)
 q=[d for d in c if d.name.split('/')[-1].lower()=='image'];return max(q or c,key=lambda d:np.prod(d.shape))
def read_aline(p:Path):
 with h5py.File(p,'r') as f:
  d=dataset(f);shape=tuple(d.shape)
  if shape==(801,512,512):a=np.asarray(d[:,X0,Y0])
  elif shape==(512,512,801):a=np.asarray(d[Y0,X0,:])
  else:raise RuntimeError((d.name,shape))
  return a.astype(float),{'dataset':d.name,'stored_shape':list(shape),'dtype':str(d.dtype)}
def model(p,z):
 a1,m1,s1,a2,m2,s2,b=p
 return b+a1*np.exp(-.5*((z-m1)/s1)**2)+a2*np.exp(-.5*((z-m2)/s2)**2)
def main():
 root=Path(__file__).resolve().parents[1];cache=root/'.cache'/NAME;out=root/'reports'/'published_fig5_aline';out.mkdir(parents=True,exist_ok=True);download(cache);amp,meta=read_aline(cache)
 inten=np.maximum(amp,0)**2;mask=(ZSAVED_MM>=.52)&(ZSAVED_MM<=.55);z=ZSAVED_MM[mask];y=inten[mask];y/=max(y.max(),1e-30)
 # Initialize from two highest separated local peaks.
 loc=np.where((y[1:-1]>y[:-2])&(y[1:-1]>=y[2:]))[0]+1;loc=loc[np.argsort(y[loc])[::-1]];chosen=[]
 for i in loc:
  if all(abs(i-j)>=4 for j in chosen):chosen.append(int(i))
  if len(chosen)==2:break
 if len(chosen)<2:chosen=[int(np.argmax(y[:len(y)//2])),len(y)//2+int(np.argmax(y[len(y)//2:]))]
 chosen=sorted(chosen,key=lambda i:z[i]);p0=[y[chosen[0]],z[chosen[0]],.001,y[chosen[1]],z[chosen[1]],.001,max(0,float(np.percentile(y,5)))]
 lo=[0,z.min(),.00015,0,z.min(),.00015,-.2];hi=[2,z.max(),.01,2,z.max(),.01,.5]
 res=least_squares(lambda p:model(p,z)-y,p0,bounds=(lo,hi),loss='soft_l1',max_nfev=5000)
 p=res.x;order=np.argsort([p[1],p[4]]);widths=[2*np.sqrt(2*np.log(2))*p[2],2*np.sqrt(2*np.log(2))*p[5]];centers=[p[1],p[4]];widths=[widths[i]*1000 for i in order];centers=[centers[i] for i in order]
 # Visible widths at fixed dB thresholds around each fitted Gaussian, not a claim of physical FWHM.
 visible={str(db):[float(w*np.sqrt(db*np.log(10)/(10*np.log(2)))) for w in widths] for db in [20,40,60,70]}
 report={'experiment':'authors_processed_MIAA_ISAM_exact_Figure5_Aline_refit','source_file':NAME,'md5':digest(cache),'hdf5':meta,'indices_zero_based':{'x':X0,'y':Y0},'z_fit_mm':[.52,.55],'fit_success':bool(res.success),'centers_mm':centers,'fwhm_um':widths,'fit_cost':float(res.cost),'visible_gaussian_full_width_um_at_intensity_dB':visible,'paper_reported_fwhm_um':[1.8,2.1],'boundary':'Authors processed amplitude output only; confirms final displayed data and independent fit, not raw-camera preprocessing or MIAA spectral truth.'}
 (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');np.savez_compressed(out/'figure5_miaa_aline.npz',z_mm=z,intensity_normalized=y,fit=model(p,z),fit_parameters=p)
 plt.figure(figsize=(8.5,5.2));plt.plot(z,y,'o',ms=3,label='deposited samples');zz=np.linspace(z.min(),z.max(),1000);plt.plot(zz,model(p,zz),label='independent two-Gaussian fit');plt.xlabel('z (mm)');plt.ylabel('normalized intensity');plt.title('Exact published Figure-5 MIAA+ISAM A-line');plt.legend();plt.tight_layout();plt.savefig(out/'figure5_miaa_aline_fit.png',dpi=180);plt.close();cache.unlink();print(json.dumps(report,indent=2))
if __name__=='__main__':main()
