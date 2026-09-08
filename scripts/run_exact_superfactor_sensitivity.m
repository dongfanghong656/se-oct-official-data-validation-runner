% Exact-author FIAA/MIAA sensitivity to the user-selected super factor.
pkg load signal;
data_file=getenv('DATA_FILE');code_dir=getenv('UPSTREAM_CODE_DIR');out_file=getenv('OUTPUT_MAT');
if isempty(data_file)||isempty(code_dir),error('DATA_FILE and UPSTREAM_CODE_DIR required');end
if isempty(out_file),out_file='reports/exact_superfactor/results.mat';end
addpath(code_dir);load(data_file,'Cscan','sk');
coords_yx0=[12,7;232,210;482,38;1,453];coords_yx1=coords_yx0+1;
iRawDatas=flip(fft(Cscan,[],1),1);sk=flip(sk(:));b=sort(sk,'descend');mask=sk>=b(128);ii=(find(mask,1,'first'):find(mask,1,'last'))';
Nz=200;supers=[2,3,4,6,8];q_i=10;eta_weight=1;Nlines=size(coords_yx1,1);
maxK=Nz*max(supers);rfiaa=complex(nan(maxK,Nlines,length(supers)));rfmap=rfiaa;miaa=rfiaa;
for n=1:Nlines
 raw=iRawDatas(:,coords_yx1(n,1),coords_yx1(n,2));gk=(raw'*sk)/(sk'*sk);C=raw-gk*sk;anC=C./sk;demod=anC(ii)-mean(anC(ii));
 for si=1:length(supers)
  super=supers(si);K=Nz*super;[a,~,amap]=fiaa_oct_c1(demod,K,q_i,eta_weight);rfiaa(1:K,n,si)=a;rfmap(1:K,n,si)=amap;
  [~,~,map2]=fiaa_oct_c1(demod,2*K,q_i,eta_weight);fa=[map2(1);map2(end:-1:2)];xr=ifft(fa,[],1)*(2*K);shift_amount=round(Nz*((super-1)/2))+ii(1)-1;xr=circshift(xr,shift_amount,1);xr=xr(1:K);
  edge_n=min(100,floor(K/4));h=hanning(2*edge_n);w=ones(K,1);w(1:edge_n)=h(1:edge_n);w(K-edge_n+1:K)=h(edge_n+1:end);miaa(1:K,n,si)=ifft(xr.*w,K,1);
 end
end
save('-mat7-binary',out_file,'coords_yx0','supers','Nz','q_i','eta_weight','rfiaa','rfmap','miaa');fprintf('SUPERFACTOR_OK output=%s\n',out_file);
