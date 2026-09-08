#!/usr/bin/env python3
"""3-D ISAM causal validation on exact-author MIAA spectra from a public-data crop."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

DX_M = 0.225e-3/512
LAMBDA_MIN = 5.011840350737296e-7
LAMBDA_MAX = 5.25482346025955e-7
REFRACTIVE_INDEX = 1.33
FOCUS_INDEX = 90*4


def edge_taper(n:int, edge:int=100)->np.ndarray:
    w=np.ones(n,float); h=np.hanning(2*edge)
    w[:edge]=h[:edge]; w[-edge:]=h[edge:]
    return w


def lateral_mask(kx:np.ndarray,ky:np.ndarray)->np.ndarray:
    xx,yy=np.meshgrid(kx,ky,indexing='ij'); r=np.sqrt(xx*xx+yy*yy)
    inner=3.1e6; outer=5.9e6
    w=np.ones_like(r); w[r>=outer]=0
    mid=(r>inner)&(r<outer)
    w[mid]=0.5+0.5*np.cos(np.pi*(r[mid]-inner)/(outer-inner))
    return w


def interp_complex(x:np.ndarray,y:np.ndarray,xnew:np.ndarray)->np.ndarray:
    order=np.argsort(x); xs=x[order]; ys=y[order]
    keep=np.r_[True,np.diff(xs)>0]; xs=xs[keep]; ys=ys[keep]
    if xs.size<4: return np.zeros_like(xnew,dtype=np.complex128)
    rr=CubicSpline(xs,ys.real,extrapolate=False)(xnew)
    ii=CubicSpline(xs,ys.imag,extrapolate=False)(xnew)
    out=rr+1j*ii; out[~np.isfinite(out)]=0
    return out


def isam3(spectra:np.ndarray, *, phase_screen:np.ndarray|None=None, spatial_taper:bool=False)->tuple[np.ndarray,np.ndarray]:
    s=np.asarray(spectra,np.complex128).copy(); nz,nx,ny=s.shape
    if phase_screen is not None: s*=np.exp(1j*phase_screen)[None,:,:]
    s*=edge_taper(nz)[:,None,None]
    if spatial_taper:
        wx=np.hanning(nx); wy=np.hanning(ny); s*=wx[None,:,None]*wy[None,None,:]
    m=np.arange(nz)
    s*=np.exp(1j*2*np.pi*m*FOCUS_INDEX/(nz-1))[:,None,None]
    # Match the author's fftshift(fft(fftshift(.))) convention laterally.
    skxy=np.fft.fftshift(np.fft.fftshift(
        np.fft.fft(np.fft.fft(np.fft.fftshift(np.fft.fftshift(s,axes=1),axes=2),axis=1),axis=2),
        axes=1),axes=2)
    kx=np.fft.fftshift(np.fft.fftfreq(nx,d=DX_M))*2*np.pi
    ky=np.fft.fftshift(np.fft.fftfreq(ny,d=DX_M))*2*np.pi
    skxy*=lateral_mask(kx,ky)[None,:,:]
    kmin=2*np.pi/LAMBDA_MAX*REFRACTIVE_INDEX; kmax=2*np.pi/LAMBDA_MIN*REFRACTIVE_INDEX
    kc=(kmin+kmax)/2; kr=kmax-kmin; super=4
    ks=np.linspace(kc+super/2*kr,kc-super/2*kr,nz)
    # Linear kz grid matching the kx=ky=0 column, extended to the minimum mapped kz.
    step=abs(2*(ks[1]-ks[0]))
    kz_start=2*min(ks)
    kxmax=np.max(np.abs(kx)); kymax=np.max(np.abs(ky))
    kzmin=2*np.sqrt(max(min(ks)**2-(kxmax/2)**2-(kymax/2)**2,0))
    nprepend=int(np.ceil((kz_start-kzmin)/step))
    kzlin=kz_start-step*np.arange(nprepend,0,-1)
    kzlin=np.r_[kzlin,kz_start+step*np.arange(nz)]
    out=np.zeros((kzlin.size,nx,ny),np.complex128)
    for ix,kxv in enumerate(kx):
        for iy,kyv in enumerate(ky):
            rad=ks*ks-(kxv/2)**2-(kyv/2)**2
            valid=rad>0
            kz=2*np.sqrt(np.maximum(rad,0))
            pref=np.zeros_like(kz); pref[valid]=kz[valid]/np.sqrt(kz[valid]**2+kxv*kxv+kyv*kyv)
            out[:,ix,iy]=interp_complex(kz[valid],(skxy[:,ix,iy]*pref)[valid],kzlin)
    m2=np.arange(kzlin.size)
    out*=np.exp(-1j*2*np.pi*m2*FOCUS_INDEX/(nz-1))[:,None,None]
    image=np.fft.fftshift(np.fft.fftshift(
        np.fft.ifftn(np.fft.fftshift(np.fft.fftshift(out,axes=1),axes=2)),axes=1),axes=2)
    zmax=np.pi*(200-1)/(kmax-kmin)*1e3 # mm
    z_um=np.linspace(0,zmax*1000,kzlin.size)
    return image,z_um


def fwhm(profile:np.ndarray,spacing:float)->float:
    y=np.abs(profile)**2; y=y-np.percentile(y,10); y=np.maximum(y,0)
    if y.max()<=0:return float('nan')
    y/=y.max(); p=int(np.argmax(y)); l=p; r=p
    while l>0 and y[l]>=.5:l-=1
    while r<len(y)-1 and y[r]>=.5:r+=1
    if l==p or r==p:return float('nan')
    def cr(a,b):
        return a+(.5-y[a])/(y[b]-y[a]) if y[b]!=y[a] else float(a)
    return float((cr(r-1,r)-cr(l,l+1))*spacing)


def locate(image:np.ndarray,z_um:np.ndarray)->tuple[int,int,int]:
    nz,nx,ny=image.shape; cx=nx//2; cy=ny//2
    # The selected scatterer was at original z~109 and appears near 0.50 mm after the author shifts.
    zmask=np.where((z_um>430)&(z_um<570))[0]
    roi=np.abs(image[zmask,cx-10:cx+11,cy-10:cy+11])
    q=np.unravel_index(np.argmax(roi),roi.shape)
    return int(zmask[q[0]]),int(cx-10+q[1]),int(cy-10+q[2])


def metrics(image:np.ndarray,z_um:np.ndarray,peak:tuple[int,int,int])->dict[str,float|list[int]]:
    z,x,y=peak; dx_um=DX_M*1e6; dz=float(np.median(np.diff(z_um)))
    px=image[z,:,y]; py=image[z,x,:]; pz=image[:,x,y]
    amp=np.abs(image)
    zz=slice(max(0,z-12),min(image.shape[0],z+13)); xx=slice(max(0,x-12),min(image.shape[1],x+13)); yy=slice(max(0,y-12),min(image.shape[2],y+13))
    local=np.sum(amp[zz,xx,yy]**2); broad=np.sum(amp[max(0,z-40):min(image.shape[0],z+41),:,:]**2)
    return {'peak_zxy':[z,x,y],'peak_z_um':float(z_um[z]),'fwhm_x_um':fwhm(px,dx_um),'fwhm_y_um':fwhm(py,dx_um),'fwhm_z_um':fwhm(pz,dz),'local_energy_fraction':float(local/max(broad,1e-30))}


def main():
    mat=loadmat(sys.argv[1],squeeze_me=True); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
    fbw=np.asarray(mat['FBW']); miaa=np.asarray(mat['spectra_MIAA'])
    # Author DFT-to-ISAM spectrum convention.
    dft_spec=np.roll(np.fft.fft(fbw,axis=0),300,axis=0)
    dft,z=isam3(dft_spec)
    base,_=isam3(miaa)
    peak=locate(base,z)
    rng=np.random.default_rng(20260908)
    scrambled=[]
    for seed in range(8):
        theta=rng.uniform(-np.pi,np.pi,size=miaa.shape[1:])
        im,_=isam3(miaa,phase_screen=theta)
        scrambled.append(metrics(im,z,peak))
    report={'experiment':'exact_author_functions_then_3D_ISAM_on_97x97_offfocus_crop',
      'input_boundary':'phase-corrected public Cscan; no raw-camera dispersion audit',
      'crop_shape':list(miaa.shape),'dx_um':DX_M*1e6,'focus_index':FOCUS_INDEX,
      'dft_isam':metrics(dft,z,locate(dft,z)),'miaa_isam':metrics(base,z,peak),
      'phase_scrambled_miaa_isam_runs':scrambled,
      'phase_scrambled_medians':{k:float(np.nanmedian([r[k] for r in scrambled])) for k in ['fwhm_x_um','fwhm_y_um','fwhm_z_um','local_energy_fraction']},
      'acceptance_note':'MIAA phase preservation is supported only if unperturbed MIAA+ISAM yields a localized point and random per-A-line phase reduces concentration without changing individual A-line magnitudes.',
      'crop_limitation':'Finite crop and spatial truncation prevent publication-level equivalence to the full 512x512 ISAM volume.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    z0,x0,y0=peak; coord=(np.arange(miaa.shape[1])-miaa.shape[1]//2)*DX_M*1e6
    plt.figure(figsize=(10,5.5))
    for im,label in [(dft,'DFT+ISAM'),(base,'MIAA+ISAM')]:
        p=np.abs(im[z0,:,y0])**2; p/=max(p.max(),1e-30); plt.plot(coord,p,label=label)
    plt.xlabel('x (µm)');plt.ylabel('normalized intensity');plt.title('Official off-focus point: 3-D crop ISAM');plt.legend();plt.tight_layout();plt.savefig(out/'lateral_profiles.png',dpi=180);plt.close()
    # Show unperturbed and first phase-scrambled xz sections on separate normalized dB scales.
    theta=np.random.default_rng(0).uniform(-np.pi,np.pi,size=miaa.shape[1:]); scr,_=isam3(miaa,phase_screen=theta)
    for im,name in [(base,'miaa_isam'),(scr,'phase_scrambled_miaa_isam')]:
        sec=np.abs(im[:, :, y0]); sec=20*np.log10(sec/max(sec.max(),1e-30)+1e-12)
        plt.figure(figsize=(8,5));plt.imshow(sec,aspect='auto',origin='lower',extent=[coord[0],coord[-1],z[0],z[-1]],vmin=-50,vmax=0)
        plt.xlim(-20,20);plt.ylim(z[z0]-50,z[z0]+50);plt.xlabel('x (µm)');plt.ylabel('z (µm)');plt.title(name);plt.colorbar(label='dB');plt.tight_layout();plt.savefig(out/f'{name}_xz.png',dpi=180);plt.close()
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
