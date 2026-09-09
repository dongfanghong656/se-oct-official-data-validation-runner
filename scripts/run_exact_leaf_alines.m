% Exact deposited FIAA/MIAA routines on deterministic high/mid/low-score leaf A-lines.
% Public input is already phase corrected and is not raw camera data.
pkg load signal;
data_file=getenv('DATA_FILE'); code_dir=getenv('UPSTREAM_CODE_DIR'); out_file=getenv('OUTPUT_MAT');
if isempty(data_file)||isempty(code_dir),error('DATA_FILE and UPSTREAM_CODE_DIR required');end
if isempty(out_file),out_file='reports/exact_leaf/leaf_alines.mat';end
addpath(code_dir); load(data_file,'Cscan','sk');
if ~isequal(size(Cscan),[200,512,512]),error('unexpected leaf Cscan shape');end
Nz=200;K=800;q_i=10;eta_weight=1.0;
% Author datasetno=2 convention: flip fft(Cscan), do NOT flip sk.
iRawDatas=flip(fft(Cscan,[],1),1); sk=sk(:);
b=sort(sk,'descend');mask=sk>=b(128);ii=(find(mask,1,'first'):find(mask,1,'last'))';
if length(ii)~=128,error('expected contiguous 128 source bins');end
% Deterministic score stratification on a 16-pixel grid; avoid selecting only isolated maxima.
amp=squeeze(max(abs(Cscan),[],1));
coords=[];scores=[];groups=[];
for gy=9:32:505
  for gx=9:32:505
    coords=[coords;gy,gx]; scores=[scores;amp(gy,gx)];
  end
end
[~,ord]=sort(scores);
sel=[ord(1:8);ord(round(linspace(round(numel(ord)*0.35),round(numel(ord)*0.65),8)));ord(end-7:end)];
coords_yx1=coords(sel,:); selection_score=scores(sel); group_id=[ones(8,1);2*ones(8,1);3*ones(8,1)];
Nlines=size(coords_yx1,1);
raw_spectrum=complex(zeros(Nz,Nlines)); normalized_full=complex(zeros(Nz,Nlines)); demod_spectrum=complex(zeros(128,Nlines));
dft_profile=complex(zeros(K,Nlines)); normalized_dft_profile=complex(zeros(K,Nlines)); rfiaa_profile=complex(zeros(K,Nlines)); rfiaa_map_profile=complex(zeros(K,Nlines)); miaa_spectrum=complex(zeros(K,Nlines)); miaa_profile=complex(zeros(K,Nlines));
hanning_edge=hanning(200);edge_apod=ones(K,1);edge_apod(1:100)=hanning_edge(1:100);edge_apod(K-99:K)=hanning_edge(101:200);
for n=1:Nlines
 y=coords_yx1(n,1);x=coords_yx1(n,2);raw=iRawDatas(:,y,x);gk=(raw'*sk)/(sk'*sk);C=raw-gk*sk;
 raw_spectrum(:,n)=C;dft_profile(:,n)=ifft(detrend(C),K,1);anC=C./sk;normalized_full(:,n)=anC;normalized_dft_profile(:,n)=ifft(anC-mean(anC),K,1);demod=anC(ii)-mean(anC(ii));demod_spectrum(:,n)=demod;
 [rf,~,rfmap]=fiaa_oct_c1(demod,K,q_i,eta_weight);rfiaa_profile(:,n)=rf;rfiaa_map_profile(:,n)=rfmap;
 [~,~,map2]=fiaa_oct_c1(demod,2*K,q_i,eta_weight);fa=[map2(1);map2(end:-1:2)];xr=ifft(fa,[],1)*(2*K);shift_amount=round(Nz*((4-1)/2))+ii(1)-1;xr=circshift(xr,shift_amount,1);xr=xr(1:K);miaa_spectrum(:,n)=xr;miaa_profile(:,n)=ifft(xr.*edge_apod,K,1);
 fprintf('leaf line %d/%d group=%d score=%g\n',n,Nlines,group_id(n),selection_score(n));
end
coords_yx0=coords_yx1-1;
save('-mat7-binary',out_file,'coords_yx0','coords_yx1','selection_score','group_id','ii','sk','K','Nz','raw_spectrum','normalized_full','demod_spectrum','dft_profile','normalized_dft_profile','rfiaa_profile','rfiaa_map_profile','miaa_spectrum','miaa_profile','edge_apod');
fprintf('EXACT_LEAF_OK n=%d output=%s\n',Nlines,out_file);
