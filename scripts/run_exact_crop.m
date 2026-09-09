% Exact deposited FIAA/RFIAA/MIAA functions on a reusable 97x97 TiO2 crop.
% Input is the authors' phase-corrected complex C-scan crop, not raw camera spectra.
pkg load signal;

data_file = getenv('DATA_FILE');
code_dir = getenv('UPSTREAM_CODE_DIR');
out_file = getenv('OUTPUT_MAT');
if isempty(data_file) || isempty(code_dir), error('DATA_FILE and UPSTREAM_CODE_DIR required'); end
if isempty(out_file), out_file = 'reports/exact_crop/crop.mat'; end
addpath(code_dir);
loaded = load(data_file);
if isfield(loaded,'Ccrop')
  Ccrop = loaded.Ccrop;
elseif isfield(loaded,'Cscan')
  Cscan = loaded.Cscan;
  y_center0 = 232; x_center0 = 210; half = 48;
  Ccrop = Cscan(:,y_center0-half+1:y_center0+half+1,x_center0-half+1:x_center0+half+1);
else
  error('Input has neither Ccrop nor Cscan');
end
if ~isfield(loaded,'sk'), error('Input has no sk'); end
sk = loaded.sk;

Nz = size(Ccrop,1); super = 4; K = Nz*super; q_i = 10; q_rci = 2; eta_weight = 1.0;
[~,Nx,Ny] = size(Ccrop);
if Nx ~= 97 || Ny ~= 97, error('Expected 97x97 crop, got %dx%d',Nx,Ny); end

iRaw = flip(fft(Ccrop,[],1),1);
sk = flip(sk(:));
b = sort(sk,'descend'); mask = sk>=b(128);
ii = (find(mask,1,'first'):find(mask,1,'last'))';
if length(ii) ~= 128, error('Expected contiguous 128-bin source window'); end

FBW = complex(zeros(K,Nx,Ny,'single'));
spectra_MIAA = complex(zeros(K,Nx,Ny,'single'));
shift_amount = round(Nz*((super-1)/2)) + ii(1) - 1;
chunk_edges = round(linspace(1,Nx+1,5));

for jy = 1:Ny
  iRawData = iRaw(:,:,jy);
  C = complex(zeros(Nz,Nx)); sk0 = sk'*sk;
  for ix = 1:Nx
    gk = (iRawData(:,ix)'*sk)/sk0;
    C(:,ix) = iRawData(:,ix)-gk*sk;
  end
  FBW(:,:,jy) = single(ifft(detrend(C),K,1));
  anC = C./sk;
  demod = anC(ii,:) - mean(anC(ii,:),1);
  maps = complex(zeros(2*K,Nx));
  for ch = 1:4
    first = chunk_edges(ch); last = chunk_edges(ch+1)-1;
    [a,PE,amap] = fiaa_oct_c1(demod(:,first),2*K,q_i,eta_weight);
    maps(:,first) = amap;
    previous_power = abs(a).^2; previous_eta = PE(q_i+1);
    for ix = first+1:last
      [a,previous_eta,amap] = rec_fiaa_oct_c1(demod(:,ix),2*K,q_rci,eta_weight,previous_power,previous_eta);
      maps(:,ix) = amap; previous_power = abs(a).^2;
    end
  end
  for ix = 1:Nx
    fa = [maps(1,ix);maps(end:-1:2,ix)];
    xr = ifft(fa,[],1)*(2*K);
    xr = circshift(xr,shift_amount,1);
    spectra_MIAA(:,ix,jy) = single(xr(1:K));
  end
  if mod(jy,10)==0, fprintf('processed y=%d/%d\n',jy,Ny); end
end

save('-mat7-binary',out_file,'FBW','spectra_MIAA','sk','ii','Nz','Nx','Ny','K','super','shift_amount');
fprintf('EXACT_CROP_OK shape=%dx%dx%d ii=%d:%d output=%s\n',K,Nx,Ny,ii(1),ii(end),out_file);
