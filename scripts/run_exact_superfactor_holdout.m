% Exact-author MIAA target-support audit on four deterministic TiO2 A-lines.
pkg load signal;
data_file=getenv('DATA_FILE');code_dir=getenv('UPSTREAM_CODE_DIR');out_file=getenv('OUTPUT_MAT');
if isempty(data_file)||isempty(code_dir),error('DATA_FILE and UPSTREAM_CODE_DIR required');end
if isempty(out_file),out_file='reports/superfactor_holdout/source.mat';end
addpath(code_dir);load(data_file,'Cscan','sk');
coords_yx0=[12,7;232,210;482,38;1,453];coords_yx1=coords_yx0+1;
Nz=200;supers=[2,3,4,6,8];q_i=10;eta_weight=1;Nlines=size(coords_yx1,1);
iRawDatas=flip(fft(Cscan,[],1),1);sk=flip(sk(:));b=sort(sk,'descend');mask=sk>=b(128);ii=(find(mask,1,'first'):find(mask,1,'last'))';
maxK=Nz*max(supers);spectra=complex(nan(maxK,Nlines,length(supers)));normalized_full=complex(zeros(Nz,Nlines));raw_spectrum=normalized_full;
for n=1:Nlines
 raw=iRawDatas(:,coords_yx1(n,1),coords_yx1(n,2));gk=(raw'*sk)/(sk'*sk);C=raw-gk*sk;raw_spectrum(:,n)=C;anC=C./sk;normalized_full(:,n)=anC;demod=anC(ii)-mean(anC(ii));
 for si=1:length(supers)
  super=supers(si);K=Nz*super;
  [~,~,map2]=fiaa_oct_c1(demod,2*K,q_i,eta_weight);
  fa=[map2(1);map2(end:-1:2)];xr=ifft(fa,[],1)*(2*K);
  shift_amount=round(Nz*((super-1)/2))+ii(1)-1;xr=circshift(xr,shift_amount,1);
  spectra(1:K,n,si)=xr(1:K);
 end
 fprintf('line %d/%d complete\n',n,Nlines);
end
save('-mat7-binary',out_file,'coords_yx0','Nz','supers','q_i','eta_weight','sk','ii','normalized_full','raw_spectrum','spectra');
fprintf('SUPERFACTOR_HOLDOUT_OK output=%s\n',out_file);
