#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import CubicSpline


def edge_taper(n:int)->np.ndarray:
    h=np.hanning(200); w=np.ones(n); w[:100]=h[:100]; w[-100:]=h[100:]; return w

def lateral_mask(n:int)->np.ndarray:
    q=np.arange(-n//2,n-n//2,dtype=float); r=np.abs(q)
    inner=110/256*(n/2); outer=210/256*(n/2)
    w=np.ones(n); mid=(r>=inner)&(r<=outer); w[r>outer]=0
    w[mid]=0.5+0.5*np.cos(np.pi*(r[mid]-inner)/(outer-inner)); return w

def interp_complex(x,y,xn):
    order=np.argsort(x); x=x[order]; y=y[order]
    keep=np.r_[True,np.diff(x)>0]; x=x[keep]; y=y[keep]
    if len(x)<4: return np.zeros_like(xn,dtype=complex)
    rr=CubicSpline(x,y.real,extrapolate=False)(xn); ii=CubicSpline(x,y.imag,extrapolate=False)(xn)
    z=rr+1j*ii; z[~np.isfinite(z)]=0; return z

def isam2_exact(spectra_kx,zf_index):
    # spectra shape K,x and follows the authors' extrapolated k ordering.
    K,Nx=spectra_kx.shape; m=np.arange(K)
    spectra=spectra_kx*edge_taper(K)[:,None]
    spectra=spectra*np.exp(1j*2*np.pi*m*zf_index/(K-1))[:,None]
    S=np.fft.fftshift(np.fft.fft(np.fft.fftshift(spectra,axes=1),axis=1),axes=1)
    S*=lateral_mask(Nx)[None,:]
    lambda_min=5.011840350737296e-7; lambda_max=5.25482346025955e-7; n=1.33
    k_min=2*np.pi/lambda_max*n; k_max=2*np.pi/lambda_min*n
    kc=(k_min+k_max)/2; kr=k_max-k_min; super=4
    kends=np.array([kc+super/2*kr,kc-super/2*kr])
    ks=np.linspace(kends[0],kends[1],K)
    dx=0.225e-3/512; fov=Nx*dx
    kx=np.linspace(-Nx/(2*fov),(Nx-2)/(2*fov),Nx)*2*np.pi
    kxx=np.broadcast_to(kx[None,:],(K,Nx)); kks=np.broadcast_to(ks[:,None],(K,Nx))
    rad=kks**2-(kxx/2)**2; rad=np.maximum(rad,0); kzz=2*np.sqrt(rad)
    kzcenter=kzz[:,Nx//2]
    kzmin=2*np.sqrt(max(np.min(kks)**2-(np.max(np.abs(kxx))/2)**2,0))
    kzfull=np.arange(2*K)*(kzcenter[1]-kzcenter[0])+kzcenter[0]
    kzlin=kzfull[kzfull>=kzmin]
    out=np.zeros((len(kzlin),Nx),complex)
    pref=kzz/np.sqrt(np.maximum(kzz**2+kxx**2,1e-30))
    for ix in range(Nx): out[:,ix]=interp_complex(kzz[:,ix],S[:,ix]*pref[:,ix],kzlin)
    m2=np.arange(len(kzlin)); out*=np.exp(-1j*2*np.pi*m2*zf_index/(K-1))[:,None]
    image=np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(out,axes=1)),axes=1)
    return image,kzlin,kx

def fwhm_axis(v,dx):
    y=np.abs(v)**2; p=int(np.argmax(y)); h=y[p]/2
    l=p
    while l>0 and y[l]>=h: l-=1
    r=p
    while r<len(y)-1 and y[r]>=h: r+=1
    if l==p or r==p:return float('nan')
    xl=l+(h-y[l])/(y[l+1]-y[l]); xr=(r-1)+(h-y[r-1])/(y[r]-y[r-1]); return float((xr-xl)*dx)

def local_energy(img,z,x,rz=8,rx=8):
    a=np.abs(img)**2; z0=max(0,z-rz);z1=min(a.shape[0],z+rz+1);x0=max(0,x-rx);x1=min(a.shape[1],x+rx+1)
    return float(a[z0:z1,x0:x1].sum())

def main():
    src=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
    d=loadmat(src,squeeze_me=True); K=int(d['K']); zf=90*4; Nx=d['spectra_MIAA'].shape[1]
    miaa=np.asarray(d['spectra_MIAA'],complex); fbw=np.asarray(d['FBW'],complex)
    # Authors' DFT-to-ISAM spectral preparation.
    dftspec=np.roll(np.fft.fft(fbw,axis=0),300,axis=0)
    im_m,kz,kx=isam2_exact(miaa,zf); im_d,_,_=isam2_exact(dftspec,zf)
    rng=np.random.default_rng(7); phases=rng.uniform(-np.pi,np.pi,Nx)
    im_s,_,_=isam2_exact(miaa*np.exp(1j*phases)[None,:],zf)
    # Search the physical point near the strip center; avoid wrap-around margins.
    amp=np.abs(im_m); xc=Nx//2; xsl=slice(xc-18,xc+19); zmargin=20
    sub=amp[zmargin:-zmargin,xsl]; iz,ix=np.unravel_index(np.argmax(sub),sub.shape); iz+=zmargin; ix+=xc-18
    dx=0.225e-3/512*1e6
    peak=float(amp[iz,ix]); e=local_energy(im_m,iz,ix)
    methods={}
    for name,img in [('dft_isam',im_d),('miaa_isam',im_m),('phase_scrambled_miaa_isam',im_s)]:
      # Use same target location and allow only small local peak recentering.
      a=np.abs(img); sub=a[max(0,iz-4):iz+5,max(0,ix-4):ix+5]; dz,ddx=np.unravel_index(np.argmax(sub),sub.shape)
      z=max(0,iz-4)+dz; x=max(0,ix-4)+ddx
      methods[name]={
       'peak_coordinate_zx':[int(z),int(x)],'lateral_fwhm_um':fwhm_axis(img[z,:],dx),
       'peak_amplitude_relative_to_miaa':float(a[z,x]/max(peak,1e-30)),
       'local_energy_relative_to_miaa':float(local_energy(img,z,x)/max(e,1e-30))}
    # No-ISAM lateral profile at the conventional depth peak, for reference.
    raw=np.asarray(fbw); zr,xr=np.unravel_index(np.argmax(np.abs(raw[:,xc-18:xc+19])),raw[:,xc-18:xc+19].shape); xr+=xc-18
    methods['dft_no_isam']={'peak_coordinate_zx':[int(zr),int(xr)],'lateral_fwhm_um':fwhm_axis(raw[zr,:],dx)}
    report={'experiment':'exact_recursive_MIAA_to_2D_ISAM_official_strip','input_boundary':'authors phase-corrected Cscan',
      'strip':{'y_zero_based':232,'x_zero_based':[146,274],'Nx':Nx,'dx_um':dx,'focus_index':zf},
      'target_miaa_isam_zx':[int(iz),int(ix)],'methods':methods,
      'interpretation':'Random per-A-line phase leaves each A-line magnitude unchanged but should reduce coherent ISAM concentration.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,5)); xx=(np.arange(Nx)-Nx//2)*dx
    for name,img in [('DFT+ISAM',im_d),('MIAA+ISAM',im_m),('phase-scrambled MIAA+ISAM',im_s)]: ax.plot(xx,np.abs(img[iz,:])/max(np.abs(img[iz,:]).max(),1e-30),label=name)
    ax.set_xlim(-20,20);ax.set_xlabel('lateral coordinate (µm)');ax.set_ylabel('normalized amplitude');ax.set_title('Official off-focus TiO2 strip: cross-A-line phase causal test');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'lateral_profiles.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
