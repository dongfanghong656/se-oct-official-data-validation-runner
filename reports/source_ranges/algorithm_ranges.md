# Verified upstream source ranges for independent implementation audit

Source: Zenodo record 7870795. Excerpts are preserved only for parameter/formula auditing.

## MIAA_ISAM_processing.m lines 1-39
```matlab
     1	%% Cscan_reconstruction.m
     2	% This script does RFIAA, MIAA and ISAM processing 
     3	% if saving data, datasavefolder needs to be created
     4	% 
     5	clear all
     6	close all
     7	%% choosing the datasets, the upsampling factor (super) and whether to save data after processing
     8	super=4; % super-resolution factor >=2, choose a power of 2. (M/N)
     9	save_complexdata = 0; % whether to save complex data to apply CAO afterwards (only for ISAM data) 0=no, 1=yes
    10	save_magnitudedata = 0; %whether to save magnitude data (in this way the input data for the point scatteres fitting is saved)
    11	
    12	datasetno = 3; %TiO2 particles in gelatin sample, for figure 3 
    13	% datasetno = 2; %plant leaf sample for figure 5
    14	% datasetno = 3; % simulation data, used for analysis in figure 4 (but not to plot fig 4)
    15	if datasetno == 1
    16	    load('input_TiO2gelatin_004_phasecorrected.mat','Cscan','sk') %sk is the source spectrum
    17	    % sk = mean(mean(abs(iRawDatas),3),2); %alternative way to obtain sk
    18	    iRawDatas = fft(Cscan,[],1);
    19	    iRawDatas = flip(iRawDatas,1);
    20	    sk = flip(sk);
    21	    lateral_apodization_nonISAM = 0;
    22	    datasavefolder = 'data/exp_pointscat/';
    23	    zf_index_IAA = 90*super; % focus depth index 
    24	
    25	    
    26	elseif datasetno == 2
    27	    load('input_leafdisc_phasecorrected.mat','Cscan','sk') %sk is the source spectrum
    28	    % sk = mean(mean(abs(iRawDatas),3),2); %alternative way to obtain sk
    29	    iRawDatas = fft(Cscan,[],1);
    30	    iRawDatas = flip(iRawDatas,1);
    31	%     sk = flip(sk);
    32	    lateral_apodization_nonISAM = 0;
    33	    datasavefolder = 'data/exp_leaf/';
    34	    zf_index_IAA = 62*super; % focus depth index 
    35	
    36	elseif datasetno == 3
    37	    load('input_simulations_pointscatters_SLDshape_98zf_noise75.mat','Cscan','sk')
    38	    iRawDatas = fft(Cscan,[],1);
    39	    lateral_apodization_nonISAM = 1;
```

## MIAA_ISAM_processing.m lines 40-130
```matlab
    40	    datasavefolder = 'data/simu_pointscat/';
    41	    zf_index_IAA = (98.8)*super;% simulations 48scat
    42	
    43	end
    44	    
    45	
    46	%% applying DFT with zeropadding, RFIAA and MIAA
    47	% Select part of the spectrum used for the spectral estimation, this
    48	% contains the region with the highest spectrum intensity
    49	sk_length = 128;
    50	b = sort(sk,'descend');
    51	mask = sk>=b(sk_length);
    52	ii = [find(mask,true,'first'):find(mask,true,'last')];
    53	
    54	
    55	[Nz,Nx,Ny] = size(iRawDatas); %obtain shape of data
    56	Nz_rawdata = Nz;
    57	K = Nz*super; %size of reconstruction grid (supposing full axial coverage)
    58	
    59	t=zeros(4,1); % record times of processing
    60	
    61	% pre allocate matrices for the processed data
    62	FBW_im=zeros(Nz*super,Nx,Ny);
    63	RFIAA_im=zeros(Nz*super,Nx,Ny);
    64	RFIAA_im_map=zeros(Nz*super,Nx,Ny);  
    65	spectra_MIAA=zeros(Nz*super,Nx,Ny);   
    66	
    67	tStart = tic;
    68	for jj=1:Ny
    69	    iRawData=iRawDatas(:,:,jj);
    70	    C=zeros(Nz,Nx);
    71	    % subtract weighted background spectrum, with a weighting per B-scan
    72	    sk0=sk'*sk;
    73	    for i=1:Nx
    74	        gk=(iRawData(:,i)'*sk)/sk0;
    75	        C(:,i)=iRawData(:,i)-gk*sk;
    76	    end
    77	
    78	    %% DFT processing with interpolation
    79	    detrC=detrend(C);
    80	%     DT(:,:,jj)=detrC;  %  copy and track
    81	    tic
    82	    FBW_im(:,:,jj)=(ifft(detrC,K,1));   % DFT iamge
    83	    t(1)=t(1)+toc;
    84	    
    85	    %% Spectral estimation preprocessing
    86	    % demodulation of the spectrum and selecting chosen bandwidth 'ii'
    87	    anC=C./sk; %   
    88	    demod_spectrum= (anC(ii,:))-mean(anC(ii,:)); % subtract mean
    89	    %% RFIAA and MIAA processing
    90	    IAA.q_i=10; % number of iteration
    91	    eta=1e0;
    92	    IAA.q_rci=2; % recursive iterations
    93	    IAA.L=4;    % amount of cores/data-chuncks for parallel processing. B-scan is divided into IAA.L datachuncks
    94	    IAA.recursive_yn=2; % 1:non-recursive (FIAA); 2: recursive (RFIAA) 
    95	    IAA.red_grid=0;
    96	    IAA.methods={'FIAA','RFIAA'};
    97	
    98	    IAA.K=K; %amount of datapoints in image reconstruction
    99	    IAA.spectrum=demod_spectrum; % demodulated/uniformly reshaped interference spectrum
   100	    IAA.ROI=1:Nz*super; % Region of interest that is captured
   101	    
   102	    % RFIAA
   103	    tic
   104	    [a_m, a_m_map]=oct_iaa_c1( IAA.spectrum ,IAA.q_i,IAA.K,eta,IAA.recursive_yn,IAA.q_rci,IAA.L);
   105	    t(4)=t(4)+toc;
   106	    RFIAA_im(:,:,jj)=a_m(IAA.ROI,:); % conventional RFIAA image, used in the publication
   107	    RFIAA_im_map(:,:,jj)=a_m_map(IAA.ROI,:); % RFIAA + maximum a postiori (MAP) for extra sparsity, this is often combined, but not used in this manuscript
   108	
   109	    % MIAA based on RFIAA+MAP with K twice the goal spectrum length
   110	    % here the MAP processing is used as the MIAA spectrum can be obtained
   111	    % from the FFT of the RFIAA+MAP reconstruction
   112	    [a_m, a_m_map]=oct_iaa_c1( IAA.spectrum ,IAA.q_i,2*IAA.K,eta,IAA.recursive_yn,IAA.q_rci,IAA.L);
   113	    
   114	    fa_m_map=[a_m_map(1,:);a_m_map(end:-1:2,:)]; %matrix is reorganized (flipped except for the first index)
   115	    xr2=ifft(fa_m_map,[],1)*2*IAA.K; % IFFT of the MAP reconstruction gives the MIAA spectrum
   116	    xr2a=circshift(xr2,round(Nz*((super-1)/2))+ii(1)-1,1); %circshift brings the wavenumbers smaller than the input band to the front
   117	    xr2a=xr2a(1:Nz*super,:); % the goal spectrum length is selected
   118	
   119	    spectra_MIAA(:,:,jj)=xr2a;  % The k-extrapolated data in the range of -199:600 with reference to the original 1:400
   120	    
   121	    if mod(jj+1,20)==0
   122	        tFrames = toc(tStart);
   123	        fprintf('processing frame %d to %d took %.2f s\n',jj-19,jj,tFrames)
   124	        tStart = tic;
   125	    end
   126	end
   127	%% ISAM inverse scattering
   128	Nz_calc = Nz_rawdata;
   129	
   130	% define the spectral limits to obtain absolute k-values for the spectrum
```

## MIAA_ISAM_processing.m lines 127-260
```matlab
   127	%% ISAM inverse scattering
   128	Nz_calc = Nz_rawdata;
   129	
   130	% define the spectral limits to obtain absolute k-values for the spectrum
   131	lambda_min= 5.011840350737296e-07; %#m
   132	lambda_max= 5.25482346025955e-07; %#m
   133	lambda_center = (lambda_min + lambda_max)/2;
   134	
   135	% calculation of boundary k values
   136	refr_index = 1.33;
   137	k_min=2*pi/lambda_max*refr_index;
   138	k_max=2*pi/lambda_min*refr_index;
   139	k_minmax = [k_min,k_max];
   140	k_center = mean(k_minmax);
   141	k_range = (k_minmax(2)-k_minmax(1));
   142	k_minmax_iaa = k_center+[+super/2*k_range,-super/2*k_range]; % define k-range extrapolated MIAA spectrum
   143	
   144	% define lateral grid
   145	sizeX = 0.225;
   146	sizeY = 0.225;
   147	FOV_xy = [sizeX*1e-3,sizeY*1e-3]; 
   148	% define z range of the image based on k-values
   149	sizeZ = pi*(Nz_calc-1)/(k_minmax(2)-k_minmax(1))*1e3;
   150	%% prepare edge apodization in axial and lateral direction to reduce side-lobes
   151	%axial apodization
   152	NzIAA = length(spectra_MIAA(:,1,1));
   153	N_hannedge_length = 200; % width of hannin window that is split and put at the edges
   154	hanning_edge = hanning(N_hannedge_length);
   155	% build edge apodization
   156	edge_apod = ones(NzIAA,1);
   157	edge_apod(1:round(N_hannedge_length/2)) = hanning_edge(1:round(N_hannedge_length/2)); 
   158	edge_apod((NzIAA-round(N_hannedge_length/2)+1):end) = hanning_edge(round(N_hannedge_length/2)+1:end);
   159	
   160	%lateral apodization to reduce side-lobes and noise
   161	% radii_smoothedge = [110,256];
   162	radii_smoothedge = [110,210]; %inner and outer radius of cos^2 shaped smoothing in pixels
   163	xi = -Nx/2:Nx/2-1;
   164	[xxi,yyi] = meshgrid(xi,xi);
   165	radius = sqrt(xxi.^2+yyi.^2);
   166	mask = (radius<radii_smoothedge(1))+(radius>=radii_smoothedge(1)).*(0.5+0.5*cos(1*pi*(radius-radii_smoothedge(1))./(radii_smoothedge(2)-radii_smoothedge(1)))).*(radius<=radii_smoothedge(2));
   167	mask = reshape(mask,[1,Nx,Nx]);
   168	%% define grids for ISAM
   169	% define lateral spatial frequencies
   170	kx=linspace(-Nx/(2*FOV_xy(1)),(Nx-2)/(2*FOV_xy(1)),Nx)*2*pi;
   171	ky=linspace(-Ny/(2*FOV_xy(2)),(Ny-2)/(2*FOV_xy(2)),Ny)*2*pi;
   172	ksamp=linspace(k_minmax_iaa(1),k_minmax_iaa(2),NzIAA);
   173	m = 0:NzIAA-1;
   174	[kxx,kksamp] = meshgrid(kx,ksamp);
   175	kzz = 2*sqrt(kksamp.^2-(kxx/2).^2);
   176	%define the linear grid on which all data is interpolated
   177	kzlin = kzz(:,round(Nx/2)+1); 
   178	kzmin = 2*sqrt(min(kksamp(:)).^2-2*(max(kxx(:))/2).^2);
   179	% extend the linear grid to include all data that is mapped to lower kz
   180	kzlin=(0:2*NzIAA-1)*(kzlin(2)-kzlin(1))+kzlin(1);
   181	kzlin = kzlin(kzlin>=kzmin);
   182	
   183	%% apply ISAM on MIAA data
   184	% apply edge apodization to reduce sidelobes when zero-padding (due to interpolation)
   185	spectra = spectra_MIAA.*edge_apod; 
   186	%lateral fft
   187	S_k_Qx_Qy=fftshift(fftshift(fft(fft(fftshift(fftshift(spectra.*exp(1j*2*pi*m'*zf_index_IAA/(NzIAA-1)),2),3),[],2),[],3),2),3);
   188	% clear spectra to free space in memory
   189	clear spectra 
   190	% apply lateral apodization mask on the data
   191	S_k_Qx_Qy = S_k_Qx_Qy.*mask;
   192	%preallocate matrices for interpolated data and single frame of
   193	%interpolated data:
   194	S_Qz_Qx_Qy = zeros([length(kzlin),Nx,Ny]);
   195	Sx_Qz_Qx_Qy = zeros([length(kzlin),Nx]);
   196	tStart = tic;
   197	for j =1:Ny
   198	    %get ky value for this frame
   199	    kyv=ky(j); 
   200	    % calculate the kz values for the frame
   201	    kzz = 2*sqrt(kksamp.^2-(kxx/2).^2-(kyv/2).^2);
   202	    %apply prefactor belonging to change of variables
   203	    Sx_k_Qx_Qy = S_k_Qx_Qy(:,:,j).*kzz./sqrt(kzz.^2+kxx.^2+kyv.^2);
   204	    %iterate over kx values
   205	    parfor i = 1:Nx
   206	        % interpolate on new linear grid
   207	        Sx_Qz_Qx_Qy(:,i) = interp1(kzz(:,i),Sx_k_Qx_Qy(:,i),kzlin,'spline',0);        
   208	    end
   209	    % add frame to matrix
   210	    S_Qz_Qx_Qy(:,:,j) = Sx_Qz_Qx_Qy;
   211	    
   212	    % loop to keep track of progress
   213	    if mod(j+1,20)==0
   214	        tFrames = toc(tStart);
   215	        fprintf('ISAM interpolating MIAA data frame %d to %d took %.2f s\n',j-19,j,tFrames)
   216	        tStart = tic;
   217	    end
   218	end
   219	
   220	% define new series of integers with interpolated k length for applying the
   221	% shift
   222	m2=0:length(kzlin)-1;
   223	clear S_k_Qx_Qy; % take the large matrix out of memory
   224	
   225	% shift focus back to the original position and apply ifft to obtain the ISAM image
   226	ISAM_image=fftshift(fftshift(ifftn(fftshift(fftshift(S_Qz_Qx_Qy.*exp(-1j*2*pi*m2'*zf_index_IAA/(NzIAA-1)),2),3)),2),3);
   227	clear S_Qz_Qx_Qy % take frequency domain data out of memory
   228	%% save complex data for computational adaptive optics if needed
   229	if save_complexdata == 1
   230	    ISAM_image = flip(ISAM_image,1);
   231	    ISAM_image = ISAM_image(200:1000,:,:);
   232	    save([datasavefolder,'MIAA_ISAM_complex.mat'],'ISAM_image','-v7.3')
   233	end
   234	
   235	%% obtain the dB-compressed image for plotting (and to reduce the memory load)
   236	ISAM_imagedB = abs(ISAM_image);
   237	clear ISAM_image
   238	ISAM_imagedB = 20*log10(ISAM_imagedB/max(ISAM_imagedB(:)));
   239	
   240	%% apply ISAM to FBW data - this takes the same steps as the MIAA spectra.
   241	spectra = circshift(fft(FBW_im,[],1),300,1); 
   242	S_k_Qx_Qy=fftshift(fftshift(fft(fft(fftshift(fftshift(spectra.*exp(1j*2*pi*m'*zf_index_IAA/(NzIAA-1)),2),3),[],2),[],3),2),3);
   243	clear spectra
   244	S_k_Qx_Qy = S_k_Qx_Qy.*mask; %apply lateral apodization
   245	
   246	S_Qz_Qx_Qy = zeros([length(kzlin),Nx,Ny]);
   247	Sx_Qz_Qx_Qy = zeros([length(kzlin),Nx]);
   248	tStart = tic;
   249	for j =1:Ny
   250	    kyv=ky(j);
   251	    kzz = 2*sqrt(kksamp.^2-(kxx/2).^2-(kyv/2).^2);
   252	    Sx_k_Qx_Qy = S_k_Qx_Qy(:,:,j).*kzz./sqrt(kzz.^2+kxx.^2+kyv.^2);
   253	    %iterate over kx values
   254	    parfor i = 1:Nx
   255	        Sx_Qz_Qx_Qy(:,i) = interp1(kzz(:,i),Sx_k_Qx_Qy(:,i),kzlin,'spline',0);        
   256	    end
   257	    S_Qz_Qx_Qy(:,:,j) = Sx_Qz_Qx_Qy;
   258	    if mod(j+1,20)==0
   259	        tFrames = toc(tStart);
   260	        fprintf('ISAM interpolating DFT data frame %d to %d took %.2f s\n',j-19,j,tFrames)
```

## MIAA_ISAM_processing.m lines 261-420
```matlab
   261	        tStart = tic;
   262	    end
   263	end
   264	
   265	clear S_k_Qx_Qy; % take the large matrix out of memory
   266	ISAM_image_FBW=fftshift(fftshift(ifftn(fftshift(fftshift(S_Qz_Qx_Qy.*exp(-1j*2*pi*m2'*zf_index_IAA/(NzIAA-1)),2),3)),2),3);
   267	clear S_Qz_Qx_Qy;
   268	%% save complex data for applying computational adaptive optics if desired
   269	if save_complexdata == 1
   270	    ISAM_image_FBW = flip(ISAM_image_FBW,1);
   271	    ISAM_image_FBW = ISAM_image_FBW(200:1000,:,:);
   272	    save([datasavefolder,'DFT_ISAM_complex.mat'],'ISAM_image_FBW','-v7.3')
   273	end
   274	
   275	%% transform to dB-compressed image for displaying and to clear part of the memory
   276	ISAM_imagedB_FBW = abs(ISAM_image_FBW);
   277	clear ISAM_image_FBW
   278	ISAM_imagedB_FBW = 20*log10(ISAM_imagedB_FBW/max(ISAM_imagedB_FBW(:)));
   279	
   280	%% Obtain interpolated dB images for FBW-DFT and RFIAA such that axial sampling is consistent with ISAM datasets
   281	FBW_dB_interpolated = abs(ifft(fft(FBW_im,[],1).*edge_apod,length(ISAM_imagedB(:,1,1)),1));
   282	FBW_dB_interpolated = 20*log10(FBW_dB_interpolated/max(FBW_dB_interpolated(:)));
   283	%
   284	RFIAA_dB_interpolated = abs(ifft(fft(RFIAA_im,[],1).*edge_apod,length(ISAM_imagedB(:,1,1)),1));
   285	RFIAA_dB_interpolated = 20*log10(RFIAA_dB_interpolated/max(RFIAA_dB_interpolated(:)));
   286	%% Flip the images for the right orientation
   287	images = {FBW_dB_interpolated,RFIAA_dB_interpolated,ISAM_imagedB_FBW,ISAM_imagedB};
   288	for i=1:4
   289	    images{i} = flip(images{i},1);
   290	end
   291	%% Save image amplitude for a region of interest for fitting of the point scatterers
   292	fnames = {'DFT','RFIAA','DFT_ISAM','MIAA_ISAM'};
   293	if save_magnitudedata == 1
   294	    for i =1:4
   295	        image = 10.^(images{i}(200:1000,:,:)/20);
   296	        save([datasavefolder,'image_',fnames{i},'.mat'],'image','-v7.3')
   297	    end
   298	end
   299	%% calculate the z-grid for the ISAM images and the roi for imagesc plotting
   300	Nisam = length(ISAM_imagedB(:,1,1));
   301	zmax = pi*(Nz_calc-1)/(k_minmax(2)-k_minmax(1))*1e3; %mm
   302	z_isam = linspace(0,zmax,Nisam);
   303	Xroi = [0,sizeX];
   304	Zroi = [z_isam(1),z_isam(end)];
   305	%% plot the figures in the manuscript
   306	if datasetno ==1
   307	    run('plot_figure3.m')
   308	elseif datasetno == 2 
   309	    run('plot_figure5.m')
   310	elseif datasetno == 3
   311	    run('plot_simulationdatafigure.m')
   312	end
```

## rfiaa_oct_c1.m
```matlab
     1	    function [pS,pSmap]=rfiaa_oct_c1(pX,K,q_i,eta,q_rec)
     2	    % this fuction applies the recursive version of FIAA, RFIAA
     3	    % pX => the uniformly reshaped interference spectrum
     4	    % K => the length of the reconstruction grid in depth direction
     5	    % q_i => the number of iterations
     6	    % eta => a weighting of the noise matrix in R, default is 1
     7	    % q_rec => the number of iteration of recursive s
     8	    % pS <= the estimated a(m)
     9	    [~,piN]=size(pX);
    10	    pS=zeros(K,piN);
    11	    pSmap=pS;
    12	    [pS(:,1), PE, pSmap(:,1) ]= fiaa_oct_c1(pX(:,1),K,q_i,eta);
    13	    pee=PE(q_i+1);
    14	    for i=2:piN
    15	      [pS(:,i), pee, pSmap(:,i)]= ...
    16	          rec_fiaa_oct_c1(pX(:,i),K,q_rec,eta,abs(pS(:,i-1)).^2,pee);
    17	%       rec_fiaa_oct(pX(:,i),K,q_rec,eta,abs(pS(:,i-1)),pee);
    18	    end
    19	   
    20	    
    21	   
    22	    
    23	    
    24	    
```

## rec_fiaa_oct_c1.m lines 1-120
```matlab
     1	function [f,eta,fmap]=rec_fiaa_oct_c1(x,K,q_i,vt,f_in,eta_in)
     2	%
     3	% Recursive fast IAA
     4	%
     5	% x => Input data vector
     6	% K => Number of frequency points
     7	% q_i => Number of IAA iterations
     8	% vt => noise weighting factor (1 is default)
     9	% f_in => IAA reconstruction (a(m)) of previous scanline
    10	% eta_in => noise of previous scanline
    11	% f <= IAA reconstruction (a(m))
    12	% eta <= noise vector
    13	
    14	N=length(x);
    15	eta= eta_in;
    16	diaaf=f_in;
    17	
    18	
    19	for k=1:q_i
    20	    q=(ifft(diaaf))*K;   
    21	    c=q(1:N)+vt*eta*[1;zeros(N-1,1)]; 
    22	    [A,af]=levinson(c);
    23	    A=A.'/sqrt(af);
    24	    y=tvec_gs_i(A,x);
    25	    diaa_num=fft([y;zeros(K-N,1)]);
    26	    fa1=coeff_gs(A);
    27	    Fa=fft( fa1,K);
    28	    diaa_den=[Fa(1);Fa(end:-1:2)];
    29	    dff=diaa_num./diaa_den;
    30	    diaaf=abs(dff).^2;  
    31	    aa=A; bb=conj(A(end:-1:1)); bb=[0;bb(1:end-1)];
    32	    diag_a=cumsum(abs(aa).^2)-cumsum(abs(bb).^2);
    33	    eta= mean( abs(y./diag_a).^2 );
    34	 end
    35	   
    36	% f=diaaf;
    37	f=dff;
    38	   
    39	% MAP
    40	 q=(ifft(diaaf))*K;   
    41	    c=q(1:N)+vt*eta*[1;zeros(N-1,1)]; 
    42	    [A,af]=levinson(c);
    43	    A=A.'/sqrt(af);
    44	    y=tvec_gs_i(A,x);
    45	    diaa_num=fft([y;zeros(K-N,1)]);
    46	    fmap=diaaf.*diaa_num;
    47	
    48	
    49	function [y,q]=tvec_gs_i(a,x)
    50	%
    51	% inverse Toeplitz times vector
    52	% using GS form  
    53	%
    54	m=length(a);
    55	
    56	z=a;
    57	t1=[z;z(1);complex(zeros(m-1,1))];
    58	t1t=[conj(z(1));complex(zeros(m-1,1));conj(z(1));conj(z(end:-1:2))];
    59	
    60	w=[x;complex(zeros(m,1))];
    61	tmp=ifft(fft(t1t).*fft(w));
    62	y11=tmp(1:m);
    63	w=[y11;complex(zeros(m,1))];
    64	tmp=ifft(fft(t1).*fft(w));
    65	y1=tmp(1:m); 
    66	z=[complex(0);conj(a(end:-1:2))];
    67	t1=[z;z(1);complex(zeros(m-1,1))];
    68	t1t=[conj(z(1));complex(zeros(m-1,1));conj(z(1));conj(z(end:-1:2))];
    69	w=[x;complex(zeros(m,1))];
    70	tmp=ifft(fft(t1t).*fft(w));
    71	y12=tmp(1:m);
    72	w=[y12;complex(zeros(m,1))];
    73	tmp=ifft(fft(t1).*fft(w));
    74	y2=tmp(1:m);
    75	y=y1-y2;
    76	
    77	
    78	function phi=coeff_gs(a)
    79	%
    80	% Estimation of the coefficients of the trigonomertic polynomial
    81	%
    82	m=length(a);
    83	M=[1:m]';
    84	t1=a;
    85	t2=[complex(0);conj(a(end:-1:2))];
    86	s1=conj(t1(end:-1:1)).*M;
    87	s2=conj(t2(end:-1:1)).*M;
    88	w1=[s1;complex(zeros(m,1))];
    89	t1x=[t1;t1(1);complex(zeros(m-1,1))];
    90	tmp=ifft(fft(t1x).*fft(w1));
    91	f1=tmp(1:m); 
    92	w2=[s2;complex(zeros(m,1))];
    93	t2x=[t2;t2(1);complex(zeros(m-1,1))];
    94	tmp=ifft(fft(t2x).*fft(w2));
    95	f2=tmp(1:m); 
    96	f=f1-f2;
    97	phi=[conj(f);f(end-1:-1:1)];
    98	    
```
