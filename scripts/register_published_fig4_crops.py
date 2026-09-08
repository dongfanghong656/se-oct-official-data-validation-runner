#!/usr/bin/env python3
"""Register the same local point in authors' deposited DFT+ISAM and MIAA+ISAM crops."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'reports'/'published_fig4_point'
FILES={'DFT_ISAM':DIR/'dft_isam_fig4_point_crop.npz','MIAA_ISAM':DIR/'miaa_isam_fig4_point_crop.npz'}
DX=0.439453125;DZ=0.7887831017786151

def candidates(I):
 J=I/max(float(I.max()),1e-30)
 # Suppress pixel-scale speckle before local maxima, then verify maxima in the unsmoothed cube.
 S=gaussian_filter(J,sigma=(0.8,0.8,0.8))
 M=maximum_filter(S,size=(3,5,5),mode='nearest')
 q=np.argwhere((S==M)&(J>5e-4))
 out=[]
 for p in q:
  z,x,y=map(int,p);cube=J[max(0,z-10):z+11,max(0,x-10):x+11,max(0,y-10):y+11]
  if J[z,x,y]>=np.max(cube)*0.85:out.append((z,x,y,float(J[z,x,y])))
 return sorted(out,key=lambda t:t[3],reverse=True)
def fit(I,p):
 z0,x0,y0=p;zs=slice(max(0,z0-10),min(I.shape[0],z0+11));xs=slice(max(0,x0-10),min(I.shape[1],x0+11));ys=slice(max(0,y0-10),min(I.shape[2],y0+11));V=I[zs,xs,ys];z,x,y=np.indices(V.shape,float);m=np.unravel_index(np.argmax(V),V.shape);bg=float(np.percentile(V,10));amp=float(max(V.max()-bg,1e-30));p0=[amp,*map(float,m),2.5,2.5,2.5,bg];lo=[0,0,0,0,.3,.3,.3,-np.inf];hi=[np.inf,V.shape[0]-1,V.shape[1]-1,V.shape[2]-1,20,20,20,np.inf]
 def r(a):
  A,zc,xc,yc,sz,sx,sy,b=a;return (b+A*np.exp(-.5*(((z-zc)/sz)**2+((x-xc)/sx)**2+((y-yc)/sy)**2))-V).ravel()
 res=least_squares(r,p0,bounds=(lo,hi),loss='soft_l1',max_nfev=2000);a=res.x;f=2*np.sqrt(2*np.log(2));return {'success':bool(res.success),'fwhm_z_um':float(f*a[4]*DZ),'fwhm_x_um':float(f*a[5]*DX),'fwhm_y_um':float(f*a[6]*DX),'cost':float(res.cost)}
def main():
 data={k:np.load(v)['intensity'] for k,v in FILES.items()};cs={k:candidates(v) for k,v in data.items()};center=np.asarray(data['DFT_ISAM'].shape)//2
 matches=[]
 for a in cs['DFT_ISAM']:
  for b in cs['MIAA_ISAM']:
   da=np.asarray(a[:3]);db=np.asarray(b[:3]);delta=db-da
   # Authors' comparison allows axial +/-1; add a one-pixel lateral tolerance for independent maxima discretization.
   if abs(delta[0])<=1 and abs(delta[1])<=1 and abs(delta[2])<=1:
    dist=float(np.linalg.norm(((da+db)/2-center)*np.asarray([DZ,DX,DX])))
    matches.append((dist,a,b))
 matches.sort(key=lambda t:t[0])
 report={'experiment':'same-point_registration_in_authors_processed_full_volume_crops','crop_shape':list(data['DFT_ISAM'].shape),'candidate_counts':{k:len(v) for k,v in cs.items()},'center_local_zxy':center.tolist(),'matches_within_one_pixel':len(matches),'candidate_preview':{k:[list(x) for x in v[:12]] for k,v in cs.items()},'boundary':'Local registered reanalysis of authors final amplitude outputs; does not validate unmeasured spectrum.'}
 if matches:
  _,a,b=matches[0];report['selected']={'DFT_ISAM_local_zxy':list(a[:3]),'MIAA_ISAM_local_zxy':list(b[:3]),'physical_distance_from_crop_center_um':matches[0][0],'DFT_ISAM_fit':fit(data['DFT_ISAM'],a[:3]),'MIAA_ISAM_fit':fit(data['MIAA_ISAM'],b[:3])}
  plt.figure(figsize=(8.5,5.2))
  for name,p in [('DFT_ISAM',a[:3]),('MIAA_ISAM',b[:3])]:
   z,x,y=p;prof=data[name][:,x,y];prof=prof/max(float(prof.max()),1e-30);coord=(np.arange(len(prof))-z)*DZ;plt.plot(coord,prof,label=name)
  plt.xlim(-15,15);plt.xlabel('z relative to registered point (µm)');plt.ylabel('normalized intensity');plt.title('Same registered point in authors processed outputs');plt.legend();plt.tight_layout();plt.savefig(DIR/'registered_same_point_axial.png',dpi=180);plt.close()
 else: report['selected']=None
 (DIR/'registered_same_point_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
