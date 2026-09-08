# Upstream code audit

Files were downloaded from Zenodo record 7870795 and verified against the record MD5 values.
The snippets below are configuration-oriented extracts, not a redistributed source snapshot.

## `MIAA_ISAM_processing.m`

MD5: `8b32839402ec0b45df0a8c7cd2ea170d`

```matlab
    2: % This script does RFIAA, MIAA and ISAM processing
    9: save_complexdata = 0; % whether to save complex data to apply CAO afterwards (only for ISAM data) 0=no, 1=yes
   18: iRawDatas = fft(Cscan,[],1);
   21: lateral_apodization_nonISAM = 0;
   29: iRawDatas = fft(Cscan,[],1);
   32: lateral_apodization_nonISAM = 0;
   38: iRawDatas = fft(Cscan,[],1);
   39: lateral_apodization_nonISAM = 1;
   46: %% applying DFT with zeropadding, RFIAA and MIAA
   49: sk_length = 128;
   63: RFIAA_im=zeros(Nz*super,Nx,Ny);
   64: RFIAA_im_map=zeros(Nz*super,Nx,Ny);
   65: spectra_MIAA=zeros(Nz*super,Nx,Ny);
   82: FBW_im(:,:,jj)=(ifft(detrC,K,1));   % DFT iamge
   89: %% RFIAA and MIAA processing
   94: IAA.recursive_yn=2; % 1:non-recursive (FIAA); 2: recursive (RFIAA)
   96: IAA.methods={'FIAA','RFIAA'};
  102: % RFIAA
  106: RFIAA_im(:,:,jj)=a_m(IAA.ROI,:); % conventional RFIAA image, used in the publication
  107: RFIAA_im_map(:,:,jj)=a_m_map(IAA.ROI,:); % RFIAA + maximum a postiori (MAP) for extra sparsity, this is often combined, but not used in this manuscript
  109: % MIAA based on RFIAA+MAP with K twice the goal spectrum length
  110: % here the MAP processing is used as the MIAA spectrum can be obtained
  111: % from the FFT of the RFIAA+MAP reconstruction
  115: xr2=ifft(fa_m_map,[],1)*2*IAA.K; % IFFT of the MAP reconstruction gives the MIAA spectrum
  119: spectra_MIAA(:,:,jj)=xr2a;  % The k-extrapolated data in the range of -199:600 with reference to the original 1:400
  127: %% ISAM inverse scattering
  142: k_minmax_iaa = k_center+[+super/2*k_range,-super/2*k_range]; % define k-range extrapolated MIAA spectrum
  150: %% prepare edge apodization in axial and lateral direction to reduce side-lobes
  151: %axial apodization
  152: NzIAA = length(spectra_MIAA(:,1,1));
  153: N_hannedge_length = 200; % width of hannin window that is split and put at the edges
  155: % build edge apodization
  156: edge_apod = ones(NzIAA,1);
  157: edge_apod(1:round(N_hannedge_length/2)) = hanning_edge(1:round(N_hannedge_length/2));
  158: edge_apod((NzIAA-round(N_hannedge_length/2)+1):end) = hanning_edge(round(N_hannedge_length/2)+1:end);
  160: %lateral apodization to reduce side-lobes and noise
  168: %% define grids for ISAM
  183: %% apply ISAM on MIAA data
  184: % apply edge apodization to reduce sidelobes when zero-padding (due to interpolation)
  185: spectra = spectra_MIAA.*edge_apod;
  186: %lateral fft
  187: S_k_Qx_Qy=fftshift(fftshift(fft(fft(fftshift(fftshift(spectra.*exp(1j*2*pi*m'*zf_index_IAA/(NzIAA-1)),2),3),[],2),[],3),2),3);
  190: % apply lateral apodization mask on the data
  215: fprintf('ISAM interpolating MIAA data frame %d to %d took %.2f s\n',j-19,j,tFrames)
  225: % shift focus back to the original position and apply ifft to obtain the ISAM image
  226: ISAM_image=fftshift(fftshift(ifftn(fftshift(fftshift(S_Qz_Qx_Qy.*exp(-1j*2*pi*m2'*zf_index_IAA/(NzIAA-1)),2),3)),2),3);
  230: ISAM_image = flip(ISAM_image,1);
  231: ISAM_image = ISAM_image(200:1000,:,:);
  232: save([datasavefolder,'MIAA_ISAM_complex.mat'],'ISAM_image','-v7.3')
  236: ISAM_imagedB = abs(ISAM_image);
  237: clear ISAM_image
  238: ISAM_imagedB = 20*log10(ISAM_imagedB/max(ISAM_imagedB(:)));
  240: %% apply ISAM to FBW data - this takes the same steps as the MIAA spectra.
  241: spectra = circshift(fft(FBW_im,[],1),300,1);
  242: S_k_Qx_Qy=fftshift(fftshift(fft(fft(fftshift(fftshift(spectra.*exp(1j*2*pi*m'*zf_index_IAA/(NzIAA-1)),2),3),[],2),[],3),2),3);
  244: S_k_Qx_Qy = S_k_Qx_Qy.*mask; %apply lateral apodization
  260: fprintf('ISAM interpolating DFT data frame %d to %d took %.2f s\n',j-19,j,tFrames)
  266: ISAM_image_FBW=fftshift(fftshift(ifftn(fftshift(fftshift(S_Qz_Qx_Qy.*exp(-1j*2*pi*m2'*zf_index_IAA/(NzIAA-1)),2),3)),2),3);
  270: ISAM_image_FBW = flip(ISAM_image_FBW,1);
  271: ISAM_image_FBW = ISAM_image_FBW(200:1000,:,:);
  272: save([datasavefolder,'DFT_ISAM_complex.mat'],'ISAM_image_FBW','-v7.3')
  276: ISAM_imagedB_FBW = abs(ISAM_image_FBW);
  277: clear ISAM_image_FBW
  278: ISAM_imagedB_FBW = 20*log10(ISAM_imagedB_FBW/max(ISAM_imagedB_FBW(:)));
  280: %% Obtain interpolated dB images for FBW-DFT and RFIAA such that axial sampling is consistent with ISAM datasets
  281: FBW_dB_interpolated = abs(ifft(fft(FBW_im,[],1).*edge_apod,length(ISAM_imagedB(:,1,1)),1));
  284: RFIAA_dB_interpolated = abs(ifft(fft(RFIAA_im,[],1).*edge_apod,length(ISAM_imagedB(:,1,1)),1));
  285: RFIAA_dB_interpolated = 20*log10(RFIAA_dB_interpolated/max(RFIAA_dB_interpolated(:)));
  287: images = {FBW_dB_interpolated,RFIAA_dB_interpolated,ISAM_imagedB_FBW,ISAM_imagedB};
  292: fnames = {'DFT','RFIAA','DFT_ISAM','MIAA_ISAM'};
  295: image = 10.^(images{i}(200:1000,:,:)/20);
  299: %% calculate the z-grid for the ISAM images and the roi for imagesc plotting
  300: Nisam = length(ISAM_imagedB(:,1,1));
  302: z_isam = linspace(0,zmax,Nisam);
  304: Zroi = [z_isam(1),z_isam(end)];
```

## `fiaa_oct_c1.m`

MD5: `4839b0a7a57a7abe73f6644e38249cc6`

```matlab
   18: diaaf=(abs(fft(x,K)/N)).^2;
   22: q=(ifft(diaaf))*K;
   27: diaa_num=fft([y;zeros(K-N,1)]);
   29: Fa=fft( fa1,K);
   43: q=(ifft(diaaf))*K;
   48: diaa_num=fft([y;zeros(K-N,1)]);
   64: tmp=ifft(fft(t1t).*fft(w));
   67: tmp=ifft(fft(t1).*fft(w));
   73: tmp=ifft(fft(t1t).*fft(w));
   76: tmp=ifft(fft(t1).*fft(w));
   93: tmp=ifft(fft(t1x).*fft(w1));
   97: tmp=ifft(fft(t2x).*fft(w2));
```

## `oct_iaa_c1.m`

MD5: `d189d45be2e5f5ac3b797fb80e7b84e3`

```matlab
   16: 'RFIAA '];
   37: [S(:,:,ip),Smap(:,:,ip)]=rfiaa_oct_c1(squeeze(bX(:,:,ip)),K,q_i,eta,q_rec);
```

## `rec_fiaa_oct_c1.m`

MD5: `08133e8bc4e62063149474cd402990d2`

```matlab
   20: q=(ifft(diaaf))*K;
   25: diaa_num=fft([y;zeros(K-N,1)]);
   27: Fa=fft( fa1,K);
   40: q=(ifft(diaaf))*K;
   45: diaa_num=fft([y;zeros(K-N,1)]);
   61: tmp=ifft(fft(t1t).*fft(w));
   64: tmp=ifft(fft(t1).*fft(w));
   70: tmp=ifft(fft(t1t).*fft(w));
   73: tmp=ifft(fft(t1).*fft(w));
   90: tmp=ifft(fft(t1x).*fft(w1));
   94: tmp=ifft(fft(t2x).*fft(w2));
```

## `rfiaa_oct_c1.m`

MD5: `d5908350abe158134ebe6f9a4dce9eba`

```matlab
    1: function [pS,pSmap]=rfiaa_oct_c1(pX,K,q_i,eta,q_rec)
    2: % this fuction applies the recursive version of FIAA, RFIAA
```

## `plot_figure5.m`

MD5: `bb5ad4827fb659e3e2d6552990829d1f`

```matlab
   27: [~,Ascan_zind] = min(abs([z_isam;z_isam]-Ascan_zlim'),[],2);
   36: methods = {'DFT','RFIAA','DFT + ISAM','MIAA + ISAM'};
   37: methods_leg = {'DFT','RFIAA',['DFT+',newline,' ISAM'],['MIAA+',newline,' ISAM']};
   41: dBlims = [60,60,60,70]; %MIAA+ISAM has higher dB limits due to increased sparsity
   50: z_Ascan = z_isam(Ascan_zind(1):Ascan_zind(2));
   74: %subfigure a and c are only plotted for ISAM processed data (3 and 4)
   79: caxis([-dBlims(i),0])
   86: plot(xlims_enface1,[1,1]*z_isam(Zindex1),'--w','LineWidth',1.0)
   87: plot(xlims_enface2,[1,1]*z_isam(Zindex2),'--g','LineWidth',1.0)
   97: caxis([-dBlims(i),0])
  115: caxis([-dBlims(i),0])
  128: caxis([-dBlims(i),0])
  144: caxis([-dBlims(i),0])
```
