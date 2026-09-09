% Build an exact deposited-code MIAA spectrum for a 129-A-line strip through an off-focus TiO2 point.
pkg load signal;
data_file=getenv('DATA_FILE'); code_dir=getenv('UPSTREAM_CODE_DIR'); out_file=getenv('OUTPUT_MAT');
if isempty(data_file)||isempty(code_dir), error('DATA_FILE and UPSTREAM_CODE_DIR required'); end
if isempty(out_file), out_file='reports/exact_strip/strip.mat'; end
addpath(code_dir);
load(data_file,'Cscan','sk');
Nz=size(Cscan,1); super=4; K=Nz*super; q_i=10; q_rci=2; eta=1.0;
y0=233; x_center=211; radius=64; xs=(x_center-radius):(x_center+radius);
Cstrip=squeeze(Cscan(:,y0,xs));
iRaw=flip(fft(Cstrip,[],1),1); sk=flip(sk(:));
b=sort(sk,'descend'); mask=sk>=b(128); ii=(find(mask,1,'first'):find(mask,1,'last'))';
C=complex(zeros(size(iRaw))); sk0=sk'*sk;
for i=1:length(xs)
  gk=(iRaw(:,i)'*sk)/sk0;
  C(:,i)=iRaw(:,i)-gk*sk;
end
detrC=detrend(C); FBW=ifft(detrC,K,1);
anC=C./sk; demod=anC(ii,:)-mean(anC(ii,:),1);
Nline=size(demod,2); maps=complex(zeros(2*K,Nline));
[a,PE,amap]=fiaa_oct_c1(demod(:,1),2*K,q_i,eta);
maps(:,1)=amap; previous_power=abs(a).^2; previous_eta=PE(q_i+1);
for i=2:Nline
  [a,previous_eta,amap]=rec_fiaa_oct_c1(demod(:,i),2*K,q_rci,eta,previous_power,previous_eta);
  maps(:,i)=amap; previous_power=abs(a).^2;
end
spectra_MIAA=complex(zeros(K,Nline)); shift_amount=round(Nz*((super-1)/2))+ii(1)-1;
for i=1:Nline
  fa=[maps(1,i);maps(end:-1:2,i)]; xr=ifft(fa,[],1)*(2*K);
  xr=circshift(xr,shift_amount,1); spectra_MIAA(:,i)=xr(1:K);
end
save('-mat7-binary',out_file,'FBW','spectra_MIAA','demod','C','sk','ii','xs','y0','x_center','radius','Nz','K','super','shift_amount');
fprintf('EXACT_STRIP_OK lines=%d ii=%d:%d K=%d output=%s\n',Nline,ii(1),ii(end),K,out_file);
