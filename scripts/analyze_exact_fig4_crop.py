#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import CubicSpline
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
DX=0.225e-3/512;LMIN=5.011840350737296e-7;LMAX=5.25482346025955e-7;NREF=1.33;FOCUS=360

def edge(n,e=100):
 w=np.ones(n);h=np.hanning(2*e);w[:e]=h[:e];w[-e:]=h[e:];return w

def interp_c(x,y,xn):
 o=np.argsort(x);x=x[o];y=y[o];keep=np.r_[True,np.diff(x)>0];x=x[keep];y=y[keep]
 if len(x)<4:return np.zeros_like(xn,dtype=complex)
 r=CubicSpline(x,y.real,extrapolate=False)(xn);q=CubicSpline(x,y.imag,extrapolate=False)(xn);z=r+1j*q;z[~np.isfinite(z)]=0;return z

def isam(s,screen=None):
 s=np.asarray(s,np.complex128).copy();nz,nx,ny=s.shape
 if screen is not None:s*=np.exp(1j*screen)[None,:,:]
 s*=edge(nz)[:,None,None]
 # Mild crop-edge taper only; full author data use no spatial crop.
 s*=tukey(nx,.2)[None,:,None]*tukey(ny,.2)[None,None,:]
 m=np.arange(nz);s*=np.exp(1j*2*np.pi*m*FOCUS/(nz-1))[:,None,None]
 S=np.fft.fftshift(np.fft.fftshift(np.fft.fft(np.fft.fft(np.fft.fftshift(np.fft.fftshift(s,axes=1),axes=2),axis=1),axis=2),axes=1),axes=2)
 fovx=nx*DX;fovy=ny*DX;kx=np.linspace(-nx/(2*fovx),(nx-2)/(2*fovx),nx)*2*np.pi;ky=np.linspace(-ny/(2*fovy),(ny-2)/(2*fovy),ny)*2*np.pi
 xx,yy=np.meshgrid(kx,ky,indexing='ij');r=np.sqrt(xx*xx+yy*yy);mask=np.ones_like(r);inner=3.1e6;outer=5.9e6;mask[r>=outer]=0;mid=(r>inner)&(r<outer);mask[mid]=.5+.5*np.cos(np.pi*(r[mid]-inner)/(outer-inner));S*=mask[None,:,:]
 kmin=2*np.pi/LMAX*NREF;kmax=2*np.pi/LMIN*NREF;kc=(kmin+kmax)/2;kr=kmax-kmin;ks=np.linspace(kc+2*kr,kc-2*kr,nz)
 center=nx//2;kzz0=2*np.sqrt(ks*ks-(kx[center]/2)**2);kzmin=2*np.sqrt(min(ks)**2-(np.max(np.abs(kx))/2)**2-(np.max(np.abs(ky))/2)**2);step=kzz0[1]-kzz0[0];kzlin=np.arange(2*nz)*step+kzz0[0];kzlin=kzlin[kzlin>=kzmin]
 O=np.zeros((len(kzlin),nx,ny),complex)
 for ix,a in enumerate(kx):
  for iy,b in enumerate(ky):
   kz=2*np.sqrt(np.maximum(ks*ks-(a/2)**2-(b/2)**2,0));pref=kz/np.sqrt(kz*kz+a*a+b*b);O[:,ix,iy]=interp_c(kz,S[:,ix,iy]*pref,kzlin)
 m2=np.arange(len(kzlin));O*=np.exp(-1j*2*np.pi*m2*FOCUS/(nz-1))[:,None,None]
 im=np.fft.fftshift(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.fft.fftshift(O,axes=1),axes=2)),axes=1),axes=2)
 im=np.flip(im,axis=0)
 zmax=np.pi*(200-1)/(kmax-kmin)*1e3;z=np.linspace(0,zmax*1000,len(kzlin));return im,z

def locate(im,expected,x0,y0):
 z0=expected;rz=slice(max(0,z0-18),min(im.shape[0],z0+19));rx=slice(max(0,x0-12),min(im.shape[1],x0+13));ry=slice(max(0,y0-12),min(im.shape[2],y0+13));q=np.unravel_index(np.argmax(np.abs(im[rz,rx,ry])),np.abs(im[rz,rx,ry]).shape);return (rz.start+q[0],rx.start+q[1],ry.start+q[2])
def fwhm(p,ds):
 y=np.abs(p)**2;y=np.maximum(y-np.percentile(y,10),0);y/=max(y.max(),1e-30);i=int(np.argmax(y));l=i;r=i
 while l>0 and y[l]>=.5:l-=1
 while r<len(y)-1 and y[r]>=.5:r+=1
 if l==i or r==i:return float('nan')
 def c(a,b):return a+(.5-y[a])/(y[b]-y[a]) if y[b]!=y[a] else a
 return float((c(r-1,r)-c(l,l+1))*ds)
def base_metrics(im,z,p):
 iz,ix,iy=p;dz=float(np.median(np.diff(z)));return {'peak_zxy':[int(v) for v in p],'peak_z_um':float(z[iz]),'peak_amplitude':float(abs(im[p])),'fwhm_x_um':fwhm(im[iz,:,iy],DX*1e6),'fwhm_y_um':fwhm(im[iz,ix,:],DX*1e6),'fwhm_z_um':fwhm(im[:,ix,iy],dz)}
def local_energy(im,p,rz=5,rxy=4):
 z,x,y=p;return float(np.sum(np.abs(im[max(0,z-rz):z+rz+1,max(0,x-rxy):x+rxy+1,max(0,y-rxy):y+rxy+1])**2))

def main():
 d=loadmat(sys.argv[1],squeeze_me=True);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);miaa=np.asarray(d['spectra_MIAA']);fbw=np.asarray(d['FBW']);x0=int(d['target_local_x0']);y0=int(d['target_local_y0']);expected=int(d['expected_final_z_index_matlab1'])-1
 ds=np.roll(np.fft.fft(fbw,axis=0),300,axis=0);di,z=isam(ds);mi,_=isam(miaa);mp=locate(mi,expected,x0,y0);dp=locate(di,expected,x0,y0);bm=base_metrics(mi,z,mp);bd=base_metrics(di,z,dp);e0=local_energy(mi,mp);a0=abs(mi[mp]);rng=np.random.default_rng(20260908);runs=[]
 for seed in range(12):
  th=rng.uniform(-np.pi,np.pi,size=miaa.shape[1:]);im,_=isam(miaa,th);neigh=locate(im,mp[0],mp[1],mp[2]);runs.append({'seed':seed,'fixed_target_amplitude_ratio':float(abs(im[mp])/max(a0,1e-30)),'local_max_amplitude_ratio':float(abs(im[neigh])/max(a0,1e-30)),'fixed_local_energy_ratio':float(local_energy(im,mp)/max(e0,1e-30)),'local_max_zxy':[int(v) for v in neigh]})
 med={k:float(np.median([r[k] for r in runs])) for k in ['fixed_target_amplitude_ratio','local_max_amplitude_ratio','fixed_local_energy_ratio']}
 report={'experiment':'figure4_white_arrow_coordinate_exact_FIAA_MIAA_then_crop_ISAM','boundary':'phase-corrected public data; exact author FIAA/MIAA, finite crop and independently reimplemented ISAM','crop_shape':list(miaa.shape),'target_expected_zero_based':[expected,x0,y0],'dft_isam':bd,'miaa_isam':bm,'phase_scramble_runs':runs,'phase_scramble_medians':med,'acceptance':'Phase preservation is supported only if fixed-target amplitude and local energy decrease materially under phase scrambling; FWHM alone is not used.','limitations':['finite asymmetric crop truncates part of the defocused field','ISAM Python implementation is equation-matched but not byte-identical MATLAB spline execution']}
 (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 coordx=(np.arange(mi.shape[1])-x0)*DX*1e6
 plt.figure(figsize=(8.5,5.5))
 for im,p,label in [(di,dp,'DFT+ISAM'),(mi,mp,'MIAA+ISAM')]:
  q=np.abs(im[p[0],:,p[2]])**2;q/=max(q.max(),1e-30);plt.plot(coordx,q,label=label)
 plt.xlim(-15,15);plt.xlabel('x relative to expected target (µm)');plt.ylabel('normalized intensity');plt.title('Figure-4 out-of-focus target, coordinate-correct crop');plt.legend();plt.tight_layout();plt.savefig(out/'target_lateral_profiles.png',dpi=180);plt.close()
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
