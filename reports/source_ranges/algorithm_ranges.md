# Verified upstream source ranges for independent implementation audit

Source: Zenodo record 7870795. Excerpts are preserved only for parameter/formula auditing.

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
