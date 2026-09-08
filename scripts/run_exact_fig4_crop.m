% Exact author FIAA/RFIAA/MIAA on the coordinate-correct 181x181 Figure-4 out-of-focus crop.
pkg load signal;
data_file=getenv('DATA_FILE');code_dir=getenv('UPSTREAM_CODE_DIR');out_file=getenv('OUTPUT_MAT');
if isempty(data_file)||isempty(code_dir),error('DATA_FILE and UPSTREAM_CODE_DIR required');end
if isempty(out_file),out_file='reports/exact_fig4_crop/crop.mat';end
addpath(code_dir);s=load(data_file);Ccrop=s.Ccrop;sk=s.sk(:);
[Nz,Nx,Ny]=size(Ccrop);if Nz~=200||Nx~=181||Ny~=181,error('unexpected crop shape');end
K=800;q_i=10;q_rci=2;eta_weight=1;super=4;
iRaw=flip(fft(Ccrop,[],1),1);sk=flip(sk);b=sort(sk,'descend');mask=sk>=b(128);ii=(find(mask,1,'first'):find(mask,1,'last'))';
FBW=complex(zeros(K,Nx,Ny,'single'));spectra_MIAA=complex(zeros(K,Nx,Ny,'single'));
shift_amount=round(Nz*((super-1)/2))+ii(1)-1;chunk_edges=round(linspace(1,Nx+1,5));
for jy=1:Ny
 raw2=iRaw(:,:,jy);C=complex(zeros(Nz,Nx));sk0=sk'*sk;
 for ix=1:Nx,gk=(raw2(:,ix)'*sk)/sk0;C(:,ix)=raw2(:,ix)-gk*sk;end
 FBW(:,:,jy)=single(ifft(detrend(C),K,1));anC=C./sk;demod=anC(ii,:)-mean(anC(ii,:),1);maps=complex(zeros(2*K,Nx));
 for ch=1:4
  first=chunk_edges(ch);last=chunk_edges(ch+1)-1;[a,PE,amap]=fiaa_oct_c1(demod(:,first),2*K,q_i,eta_weight);maps(:,first)=amap;pp=abs(a).^2;pe=PE(q_i+1);
  for ix=first+1:last,[a,pe,amap]=rec_fiaa_oct_c1(demod(:,ix),2*K,q_rci,eta_weight,pp,pe);maps(:,ix)=amap;pp=abs(a).^2;end
 end
 for ix=1:Nx,fa=[maps(1,ix);maps(end:-1:2,ix)];xr=ifft(fa,[],1)*(2*K);xr=circshift(xr,shift_amount,1);spectra_MIAA(:,ix,jy)=single(xr(1:K));end
 if mod(jy,10)==0,fprintf('fig4 crop y=%d/%d\n',jy,Ny);end
end
target_local_x0=double(s.target_local_x);target_local_y0=double(s.target_local_y);expected_final_z_index_matlab1=double(s.expected_final_z_index_matlab1);
save('-mat7-binary',out_file,'FBW','spectra_MIAA','sk','ii','Nz','Nx','Ny','K','target_local_x0','target_local_y0','expected_final_z_index_matlab1');
fprintf('EXACT_FIG4_CROP_OK output=%s\n',out_file);
