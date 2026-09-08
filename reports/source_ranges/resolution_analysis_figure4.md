# resolution_analysis_figure4.m — verified source audit

```matlab
     1	% this is the script that is used for the publication figure 4
     2	
     3	clear all
     4	close all
     5	% dipimage is used for finding the point scatterer positions see
     6	% https://diplib.org/ this toolbox has to be installed
     7	% addpath('C:\Program Files\DIPimage 2.9\common\dipimage');
     8	% dip_initialise;
     9	% dipsetpref('ImageFilePath','C:\Program Files\DIPimage 2.9\images');
    10	addpath('DIPimage 2.9\common\dipimage');
    11	dip_initialise;
    12	dipsetpref('ImageFilePath','DIPimage 2.9\images');
    13	
    14	%% load experimental and simulation images for analysis
    15	load('exp_pointscat_image_DFT_ISAM.mat','image')
    16	ISAM_image{1} = image/max(image(:));
    17	load('exp_pointscat_image_MIAA_ISAM.mat','image')
    18	ISAM_image{2} = image/max(image(:));
    19	load('simu_pointscat_image_DFT_ISAM.mat','image')
    20	ISAM_image{3} = image/max(image(:));
    21	load('simu_pointscat_image_MIAA_ISAM.mat','image')
    22	ISAM_image{4} = image/max(image(:));
    23	%% set sampling distances
    24	dx = 0.225/512;
    25	dy = 0.225/512;
    26	dz = 0.7888e-3;
    27	
    28	%%
    29	zf_index = 366; % focus depth index
    30	zf_index_simu = 422; % focus depth index simulations
    31	% optimal threshold differs per dataset. this is a good combination
    32	thresholds = [1/2e3,1/2e3,1/1e4,1/2e4];
    33	% preallocate matrices
    34	coordsxy = {zeros(3,1),zeros(3,1)};
    35	coordsxz = {zeros(3,1),zeros(3,1)};
    36	volumes2 = {};
    37	Cs = {};
    38	max_volumes_all = {};
    39	noise_var = zeros(6,1);
    40	params2 = {};
    41	% allocate data for fitting
    42	half_pixwidth = 10; %half width of the fitting cube
    43	x = -half_pixwidth:half_pixwidth;
    44	y = -half_pixwidth:half_pixwidth;
    45	z = -half_pixwidth:half_pixwidth;
    46	[xx,yy,zz] = meshgrid(x,y,z);
    47	[nx,ny,nz] = size(xx);
    48	Mat(:,:,:,1)=xx;
    49	Mat(:,:,:,2)=yy;
    50	Mat(:,:,:,3)=zz;
    51	F=@(param,Mat) param(1) + param(2)*exp(-(Mat(:,:,:,1)-param(3)).^2/(param(6))^2-(Mat(:,:,:,2)-param(4)).^2/(param(7))^2-(Mat(:,:,:,3)-param(5)).^2/(param(8))^2);
    52	
    53	%process images
    54	for im_no = 1:4
    55	    threshold = thresholds(im_no);
    56	    ISAM_image_abs = (ISAM_image{im_no}/max(ISAM_image{im_no}(:))).^2;
    57	    ISAM_image_abs_orig = ISAM_image_abs; %keep original image for fitting
    58	    
    59	    % normalize data to be able to use a global threshold
    60	    depth_normalization = mean(mean(ISAM_image_abs,3),2);
    61	    depth_normalization = smoothdata(depth_normalization,'g',40);
    62	    depth_normalization = sqrt(depth_normalization/max(depth_normalization));
    63	    ISAM_image_abs = ISAM_image_abs./depth_normalization;
    64	    ISAM_image_abs = ISAM_image_abs/max(ISAM_image_abs(:));
    65	    
    66	    % get the noise variance region for SNR plot 
    67	    % (experimental (im_no<3) and simulation (im_no>3) are
    68	    % different)
    69	    if im_no<3
    70	        noise = ISAM_image_abs_orig(745:785,160:176,:);
    71	    elseif im_no>=3
    72	        noise = ISAM_image_abs_orig(100:200,50:100,:);
    73	    end
    74	    noise_var(im_no) = var(sqrt(noise(:)));    
    75	
    76	    % apply filter to find the local max values
    77	    % first local max values in enface planes
    78	    no=1;
    79	    for i = 1:length(ISAM_image_abs(:,1,1))
    80	        image = squeeze(ISAM_image_abs(i,:,:));
    81	    %     image(image<threshold)=0;
    82	        im_dip1 = dip_image(image);
    83	        im_dip2 = unif(im_dip1,4,'elliptic');
    84	        im_dip3 = unif(im_dip1,8,'elliptic');
    85	        im_dip4 = im_dip2-im_dip3;
    86	        im_dip5 = maxf(im_dip4,12,'elliptic');
    87	        im_dip6 = (im_dip5==im_dip4)*(im_dip1>threshold);
    88	        loc_max = double(im_dip6);
    89	        [rows,cols] = find(loc_max);
    90	        if length(rows)>=1
    91	            for j =1:length(rows)
    92	                coordsxy{im_no}(1,no) = i;
    93	                coordsxy{im_no}(2,no) = rows(j);
    94	                coordsxy{im_no}(3,no) = cols(j);
    95	                no = no+1;
    96	            end
    97	        end
    98	    end
    99	    %
   100	    % then local max values in x-z planes
   101	    no=1;
   102	    for i = 1:length(ISAM_image_abs(1,1,:))
   103	        image = squeeze(ISAM_image_abs(:,:,i));
   104	    %     image(image<threshold)=0;
   105	        im_dip1 = dip_image(image);
   106	        im_dip2 = unif(im_dip1,4,'elliptic');
   107	        im_dip3 = unif(im_dip1,8,'elliptic');
   108	        im_dip4 = im_dip2-im_dip3;
   109	        im_dip5 = maxf(im_dip4,12,'elliptic');
   110	        im_dip6 = (im_dip5==im_dip4)*(im_dip1>threshold);
   111	        loc_max = double(im_dip6);
   112	        [rows,cols] = find(loc_max);
   113	        if length(rows)>=1
   114	            for j =1:length(rows)
   115	                coordsxz{im_no}(3,no) = i;
   116	                coordsxz{im_no}(1,no) = rows(j);
   117	                coordsxz{im_no}(2,no) = cols(j);
   118	                no = no+1;
   119	            end
   120	        end
   121	    end
   122	
   123	    % calculate the intersection to find the local maxima in 3D 
   124	    [C,~,~] = intersect(coordsxz{im_no}',coordsxy{im_no}','rows');
   125	
   126	    % add border to image in case poin scatterers are close to the edge
   127	    [Nz,Nx,Ny] = size(ISAM_image_abs);
   128	    ISAM_image_edgeadded = zeros([Nz+2*half_pixwidth,Nx+2*half_pixwidth,Ny+2*half_pixwidth]);
   129	    ISAM_image_edgeadded(half_pixwidth+1:end-half_pixwidth,half_pixwidth+1:end-half_pixwidth,half_pixwidth+1:end-half_pixwidth) = ISAM_image_abs_orig;
   130	    
   131	    % get a small volume for each local maximum
   132	    Npeaks = length(C)
   133	    volumes = zeros(nx,ny,nz,Npeaks);
   134	    select = ones(Npeaks,1);
   135	    max_volumes = ones(Npeaks,1);
   136	    for i=1:Npeaks
   137	       xyz = C(i,:);
   138	       volume = ISAM_image_edgeadded(xyz(1)+half_pixwidth+z,xyz(2)+half_pixwidth+x,xyz(3)+half_pixwidth+y);   
   139	       max_volumes(i) = max(volume(:));       
   140	       volumes(:,:,:,i) = volume/max(volume(:));
   141	       % determine that local maximum is indeed in the center 3x3x3 pixels
   142	       % and no higher amplitude peaks in the cube
   143	       centervolume = volume(half_pixwidth:half_pixwidth+2,half_pixwidth:half_pixwidth+2,half_pixwidth:half_pixwidth+2);
   144	       if max(centervolume(:))<max(volume(:))
   145	           select(i)=0;
   146	       end
   147	    end
   148	    %select the good volumes using 'select'
   149	    volumes = volumes(:,:,:,select==1);
   150	    max_volumes =  max_volumes(select==1);
   151	    volumes2{im_no} = volumes;
   152	    Cselect = C(select==1,:);
   153	    Cs{im_no} = Cselect;
   154	    Npeaks = length(volumes(1,1,1,:));
   155	    max_volumes_all{im_no} = max_volumes;
   156	       
   157	    % fit the psfs
   158	    param0 = [0,1,0,0,0,1,2,1]; % initial parameters
   159	    params = zeros(Npeaks,length(param0));
   160	    for i = 1:Npeaks
   161	        volume = volumes(:,:,:,i);
   162	        param0 = [0,max(volume(:)),0,0,0,2,2,2]; % initial parameters
   163	        params(i,:) = lsqcurvefit(F,param0,Mat,volume); % D is the 3D matrix that needs to be fit.
   164	    end
   165	    params2{im_no} = params; %save the fit parameters
   166	end
   167	
   168	%% load simulation data without ISAM
   169	load('simu_pointscat_image_DFT.mat','image')
   170	ISAM_image{5} = image/max(image(:));
   171	load('simu_pointscat_image_RFIAA.mat','image')
   172	ISAM_image{6} = image/max(image(:));
   173	%% fit the data without ISAM on a larger volume of interest (201x201x31 pixels wide) as simulation data is sufficiently spaced
   174	half_pixwidth_lat = 100;%round((Nx-1)/2);
   175	half_pixwidthz = 15;%round((Nz-1)/2);
   176	x2 = -half_pixwidth_lat:half_pixwidth_lat;
   177	y2 = -half_pixwidth_lat:half_pixwidth_lat;
   178	z2 = -half_pixwidthz:half_pixwidthz;
   179	
   180	[xx2,zz2,yy2] = meshgrid(x2,z2,y2);
   181	[nz2,ny2,nx2] = size(xx2);
   182	Mat2 = [];
   183	Mat2(:,:,:,1)=xx2;
   184	Mat2(:,:,:,2)=yy2;
   185	Mat2(:,:,:,3)=zz2;
   186	ax_guess = [8,4]; % axial resoluiton initial guesses [DFT,RFIAA]
   187	for im_no = 5:6
   188	    ISAM_image_abs = ISAM_image{im_no}.^2;
   189	    noise = ISAM_image_abs(10:100,10:50,:);
   190	    noise_var(im_no) = var(sqrt(noise(:)));    
   191	    peaks_sel = im_no-2;
   192	    Npeaks = length(Cs{peaks_sel});
   193	    volumes = zeros(nz2,nx2,ny2,Npeaks);
   194	    max_volumes = zeros(Npeaks,1);
   195	    for i=1:Npeaks
   196	       xyz = Cs{peaks_sel}(i,:); % select scatterers based on ISAM scatterer position
   197	       volume = ISAM_image_abs(xyz(1)+z2,xyz(2)+x2,xyz(3)+y2);   
   198	       max_volumes(i) = max(volume(:));
   199	       volumes(:,:,:,i) = volume/max(volume(:));   
   200	       centervolume = volume(half_pixwidthz:half_pixwidthz+2,half_pixwidth_lat:half_pixwidth_lat+2,half_pixwidth_lat:half_pixwidth_lat+2);
   201	       if max(centervolume(:))<max(volume(:))
   202	           select(i)=0;
   203	       end
   204	    end
   205	    max_volumes_all{im_no} = max_volumes;
   206	    
   207	    param0 = [0,1,0,0,0,1,2,1]; % initial parameters
   208	    params = zeros(Npeaks,length(param0));
   209	    zR=5;
   210	    for i = 1:Npeaks
   211	        volume = volumes(:,:,:,i);
   212	        %to avoid wrong fitting, the initial guess for the lateral resolution is based on an emperically adapted theoretical formula
   213	        lat_res_guess = 0.8/(2*sqrt(log(2))*dx*1e3)*sqrt(1+((Cs{peaks_sel}(i,1)-zf_index_simu)*dz*1e3/zR).^2);
   214	        param0 = [0,max(volume(:)),0,0,0,lat_res_guess,lat_res_guess,params2{peaks_sel}(i,8)]; % initial parameters, axial guess is the fitted value with ISAM
   215	        params(i,:) = lsqcurvefit(F,param0,Mat2,volume); % D is the 3D matrix that needs to be fit.
   216	        i
   217	    end
   218	    params2{im_no} = params;
   219	end
   220	%% get physical dimensions of the fitted sigmas
   221	sigmaszxys = {};
   222	sigmaszxys_orig = {};
   223	for i=1:length(ISAM_image)
   224	sigmaszxys{i} = zeros(length(params2{i}(:,7)),3);
   225	sigmaszxys{i}(:,1) = params2{i}(:,7)*dz;
   226	sigmaszxys{i}(:,2) = params2{i}(:,6)*dx;
   227	sigmaszxys{i}(:,3) = params2{i}(:,8)*dy;
   228	sigmaszxys{i} = abs(sigmaszxys{i});
   229	sigmaszxys_orig{i} = abs(sigmaszxys{i});
   230	end
   231	
   232	%% get SNR peakfit
   233	SNRs_all_fitpeak = {};
   234	for im_no =1:6
   235	    SNRs_all_fitpeak{im_no} = params2{im_no}(:,2)./noise_var(im_no).*max_volumes_all{im_no};
   236	end
   237	
   238	%% determin where the points of DFT+ISAM and MIAA+ISAM overlap, to get a fair comparison (the same points)
   239	[C_images,ia,ib] = intersect(Cs{1},Cs{2},'rows');
   240	%with negative shift of 1 pix
   241	Cs1_n1 = Cs{1};
   242	Cs1_n1(:,1) = Cs1_n1(:,1)-1;
   243	[C_images_shiftn1,ian1,ibn1] = intersect(Cs1_n1,Cs{2},'rows');
   244	% with positive shift of 1 pix
   245	Cs1_p1 = Cs{1};
   246	Cs1_p1(:,1) = Cs1_p1(:,1)+1;
   247	[C_images_shiftp1,iap1,ibp1] = intersect(Cs1_p1,Cs{2},'rows');
   248	
   249	N_peaks = length(C_images)+length(C_images_shiftn1)+length(C_images_shiftp1);
   250	ias = cat(1,ia,ian1,iap1);
   251	ibs = cat(1,ib,ibn1,ibp1);
   252	sigmaszxy1_s2 = zeros(N_peaks,3);
   253	sigmaszxy2_s2 = zeros(N_peaks,3);
   254	C_images2 = cat(1,C_images,C_images_shiftn1,C_images_shiftp1);
   255	for i = 1:N_peaks
   256	   sigmaszxy1_s2(i,:) = sigmaszxys{1}(ias(i),:);
   257	   sigmaszxy2_s2(i,:) = sigmaszxys{2}(ibs(i),:);
   258	end
   259	
   260	%% PSF numbers for the plot (note that changes above may change the real PSF)
   261	psf_no1 = 266;
   262	psf_no2 = 84;
   263	psf_indices = zeros(2,2);
   264	psf_indices(1,:) = [ia(psf_no1),ib(psf_no1)];
   265	psf_indices(2,:) = [ia(psf_no2),ib(psf_no2)];
   266	%% load experimental images without ISAM for subplot (a) and (d)
   267	xdft = -30:30;
   268	
   269	load('exp_pointscat_image_DFT.mat','image')
   270	volumes_plot{1} = image(z+Cs{1}(psf_indices(1,1),1),x+Cs{1}(psf_indices(1,1),2),y+Cs{1}(psf_indices(1,1),3)).^2;
   271	volumes_plot{5} = image(z+Cs{1}(psf_indices(2,1),1),x+Cs{1}(psf_indices(2,1),2),y+Cs{1}(psf_indices(2,1),3)).^2;
   272	
   273	load('exp_pointscat_image_RFIAA.mat','image')
   274	volumes_plot{2} = image(z+Cs{1}(psf_indices(1,1),1),x+Cs{1}(psf_indices(1,1),2),y+Cs{1}(psf_indices(1,1),3)).^2;
   275	volumes_plot{6} = image(z+Cs{1}(psf_indices(2,1),1),x+Cs{1}(psf_indices(2,1),2),y+Cs{1}(psf_indices(2,1),3)).^2;
   276	
   277	volumes_plot{3} = volumes2{1}(:,:,:,psf_indices(1,1));
   278	volumes_plot{7} = volumes2{1}(:,:,:,psf_indices(2,1));
   279	volumes_plot{4} = volumes2{2}(:,:,:,psf_indices(1,2));
   280	volumes_plot{8} = volumes2{2}(:,:,:,psf_indices(2,2));
   281	
   282	for i=1:8
   283	    volumes_plot{i} = volumes_plot{i}/max(volumes_plot{i}(:));
   284	end
   285	%% get Gaussian function as line in subfigs (b-c,e-f)
   286	% fit the non-ISAM volumes
   287	params_noisam = zeros(4,length(param0));
   288	for i = 1:4
   289	volume = volumes_plot{i+floor(i/3)*2};
   290	param0 = [0,max(volume(:)),0,0,0,2,2,2]; % initial parameters
   291	params_noisam(i,:) = lsqcurvefit(F,param0,Mat,volume); % D is the 3D matrix that needs to be fit.
   292	end
   293	
   294	% get line from the fit with dense sampling for a smoother graph
   295	x3 = -half_pixwidth:0.1:half_pixwidth;
   296	y3 = -half_pixwidth:0.1:half_pixwidth;
   297	z3 = -half_pixwidth:0.1:half_pixwidth;
   298	[xx,yy,zz] = meshgrid(x3,y3,z3);
   299	[nx3,ny3,nz3] = size(xx);
   300	Mat3 = zeros([nx3,ny3,nz3,3]);
   301	Mat3(:,:,:,1)=xx;
   302	Mat3(:,:,:,2)=yy;
   303	Mat3(:,:,:,3)=zz;
   304	fitted_gaussians = {};
   305	fitted_gaussians{1} = F(params_noisam(1,:),Mat3);
   306	fitted_gaussians{2} = F(params_noisam(2,:),Mat3);
   307	fitted_gaussians{3} = F(params2{1}(psf_indices(1,1),:),Mat3);
   308	fitted_gaussians{4} = F(params2{2}(psf_indices(1,2),:),Mat3);
   309	fitted_gaussians{5} = F(params_noisam(3,:),Mat3);
   310	fitted_gaussians{6} = F(params_noisam(4,:),Mat3);
   311	fitted_gaussians{7} = F(params2{1}(psf_indices(2,1),:),Mat3);
   312	fitted_gaussians{8} = F(params2{2}(psf_indices(2,2),:),Mat3);
   313	half_pix2 = round((length(x2)+1)/2);
   314	
   315	%% calculate the FWHMs for the fits in subfig (b-c,e-f) so it can be used in the text
   316	FWHMs = zeros(2,4,3);
   317	for i=1:2
   318	    for j=1:2
   319	        FWHMs(i,j,:) = abs(2*sqrt(log(2))*params_noisam(2*(i-1)+j,6:8).*[dx,dz,dy]*1e3);
   320	        FWHMs(i,j+2,:) = abs(2*sqrt(log(2))*params2{j}(psf_indices(i,j),6:8).*[dx,dz,dy]*1e3);
   321	    end
   322	end
   323	
   324	%% group SNRs of experimental data in 20 um intervals
   325	delta = 20;
   326	% z_means = -180:delta:180;
   327	z_means = -200:delta:200;
   328	zpos = {};
   329	SNR_postproc = {};
   330	%DFT+ISAM
   331	zpos{1} = (C_images2(:,1)-zf_index)*dz*1e3;
   332	SNR_postproc{1} = 10*log10(SNRs_all_fitpeak{1}(ias));
   333	%MIAA+ISAM
   334	zpos{2} = (C_images2(:,1)-zf_index)*dz*1e3;
   335	SNR_postproc{2} = 10*log10(SNRs_all_fitpeak{2}(ibs));
   336	SNR_means = zeros([2,length(z_means),4]);
   337	for i = 1:length(z_means)
   338	    for j = 1:2
   339	        if i==1
   340	            mask = zpos{j}<(z_means(i)+delta/2);
   341	        elseif i == length(z_means)
   342	            mask = zpos{j}>(z_means(i)-delta/2);
   343	        else
   344	            mask = abs(zpos{j}-z_means(i))<delta/2;
   345	        end
   346	%         mask = abs(zpos{j}-z_means(i))<delta/2;
   347	        SNRs_select = SNR_postproc{j}(mask>0);
   348	        SNR_means(j,i,:) = [mean(SNRs_select),min(SNRs_select),max(SNRs_select),length(SNRs_select)];
   349	    end
   350	end
   351	
   352	%% make figure 4 of the manuscript
   353	figure(10)
   354	close(10)
   355	fig10 = figure(10);
   356	set(fig10,'Units','centimeters');
   357	set(fig10,'Position',[10,5,14.3,10.3])
   358	methods = {'DFT','RFIAA',{'DFT +','ISAM'},{'MIAA +','ISAM'}};
   359	% determine x and z limits for the A-scans
   360	xlims_alines = [-2,2];
   361	zlims_alines = [z3(1)*dz*1e3,z3(end)*dz*1e3];
   362	for i=1:4
   363	    subplot(3,6,i)
   364	    imagesc(x*dx*1e3,z*dz*1e3,squeeze(volumes_plot{i}(:,:,half_pixwidth+1)))
   365	    axis equal tight
   366	    title(methods{i})
   367	    if i==1
   368	        hold on
   369	        text(x(1)*dx*1e3+0.15*dz/dx-0.1,z(1)*dz*1e3+0.2,'\rightarrow','Color','w','HorizontalAlignment','Left','VerticalAlignment','middle','FontSize',10,'FontWeight','Bold')
   370	        text(x(1)*dx*1e3+0.15*dz/dx+2.4,z(1)*dz*1e3+0.2,'x','Color','w','HorizontalAlignment','Left','VerticalAlignment','middle','FontSize',10,'FontWeight','Bold')
   371	        text(x(1)*dx*1e3+0.3*dz/dx,z(1)*dz*1e3-0.2,'\downarrow','Color','w','HorizontalAlignment','Center','VerticalAlignment','Top','FontSize',10,'FontWeight','Bold')
   372	        text(x(1)*dx*1e3+0.3*dz/dx,z(1)*dz*1e3+2.0,'z','Color','w','HorizontalAlignment','Center','VerticalAlignment','Top','FontSize',10,'FontWeight','Bold')
   373	        plot(xlims_alines,[0,0],':w','LineWidth',1)
   374	        plot([0,0],zlims_alines,':w','LineWidth',1)
   375	    end
   376	    if i==4
   377	        hold on
   378	        plot([x(end)*dx*1e3-5.5,x(end)*dx*1e3-0.5],(z(end)*dz*1e3-0.5)*[1,1],'-w','LineWidth',2)
   379	        text(x(end)*dx*1e3-3.0,z(end)*dz*1e3-2.3,['5 ',char(181),'m'],'Color','w','HorizontalAlignment','Center','FontSize',9)
   380	    end
   381	end
   382	subplot(3,6,5)
   383	hold on
   384	colors = 'kgbr';
   385	for i=1:4
   386	    plot(x*dx*1e3,volumes_plot{i}(half_pixwidth+1,:,half_pixwidth+1),'o','Color',colors(i),'MarkerSize',3)
   387	    xlim([-2,2])
   388	end
   389	for i=1:4
   390	    plot(x3*dx*1e3,squeeze(fitted_gaussians{i}(half_pix2+1,:,half_pix2+1)),'-','Color',colors(i),'LineWidth',1)
   391	end
   392	title('I(x)')
   393	box on
   394	xlabel(['x (',char(181),'m)'])
   395	ylabel('I (norm.)')
   396	ylim([0,1.1])
   397	subplot(3,6,6)
   398	hold on
   399	
   400	for i=1:4
   401	    plot(z*dz*1e3,volumes_plot{i}(:,half_pixwidth+1,half_pixwidth+1),'o','Color',colors(i),'MarkerSize',3)
   402	end
   403	for i=1:4
   404	    plot(z3*dz*1e3,squeeze(fitted_gaussians{i}(:,half_pix2,half_pix2)),'-','Color',colors(i),'LineWidth',1)
   405	end
   406	title('I(z)')
   407	box on
   408	xlabel(['z (',char(181),'m)'])
   409	ylabel('I (norm.)')
   410	ylim([0,1.1])
   411	xlim([z3(1)*dz*1e3,z3(end)*dz*1e3])
   412	
   413	for i=5:8
   414	    subplot(3,6,i+2)
   415	    imagesc(x*dx*1e3,z*dz*1e3,squeeze(volumes_plot{i}(:,:,half_pixwidth+1)))
   416	    axis equal tight
   417	end
   418	colormap(viridis(256))
   419	subplot(3,6,11)
   420	hold on
   421	for i=1:4
   422	    plot(x*dx*1e3,volumes_plot{i+4}(half_pixwidth+1,:,half_pixwidth+1),'o','Color',colors(i),'MarkerSize',3)
   423	end
   424	for i=1:4
   425	    plot(x3*dx*1e3,squeeze(fitted_gaussians{i+4}(half_pix2,:,half_pix2)),'-','Color',colors(i),'LineWidth',1)
   426	end
   427	xlim([-2,2])
   428	ylim([0,1.1])
   429	box on
   430	xlabel(['x (',char(181),'m)'])
   431	ylabel('I (norm.)')
   432	subplot(3,6,12)
   433	methods2 = {'DFT','RFIAA','DFT + ISAM','MIAA + ISAM'};
   434	methods_leg = {'DFT','RFIAA',['DFT+',newline,' ISAM'],['MIAA+',newline,' ISAM']};
   435	hold on
   436	for i=1:4
   437	    plot(z(1)*dz*1e3-1,-1,'o-','Color',colors(i),'MarkerSize',3,'DisplayName',methods_leg{i},'LineWidth',1)
   438	end
   439	for i=1:4
   440	    plot(z*dz*1e3,volumes_plot{i+4}(:,half_pixwidth+1,half_pixwidth+1),'o','Color',colors(i),'MarkerSize',3,'HandleVisibility','off')
   441	end
   442	for i=1:4
   443	    plot(z3*dz*1e3,squeeze(fitted_gaussians{i+4}(:,half_pix2,half_pix2)),'-','Color',colors(i),'LineWidth',1,'HandleVisibility','off')
   444	end
   445	ylim([0,1.1])
   446	xlim([z3(1)*dz*1e3,z3(end)*dz*1e3])
   447	xlabel(['z (',char(181),'m)'])
   448	ylabel('I (norm.)')
   449	box on
   450	legend();
   451	
   452	%
   453	zlims = [-220,220];
   454	dzlims = [0,9.9];
   455	dxylims = [0.5,1.99];
   456	
   457	% plot SNRs (subfig (g))
   458	inset_xlim = [-30,30];
   459	ylimSNR = [0,90];
   460	subplot(3,6,13)
   461	hold on
   462	% DFT simulation
   463	plot((Cs{3}(:,1)-zf_index_simu)*dz*1e3,10*log10(SNRs_all_fitpeak{5}),'-.k','LineWidth',1.5)%
   464	% RFIAA simulation
   465	plot((Cs{4}(:,1)-zf_index_simu)*dz*1e3,10*log10(SNRs_all_fitpeak{6}),':k','LineWidth',1.5)
   466	% DFT+ISAM simulation
   467	plot((Cs{3}(:,1)-zf_index_simu)*dz*1e3,10*log10(SNRs_all_fitpeak{3}),'-k','LineWidth',1.5)
   468	% MIAA+ISAM simulation
   469	plot((Cs{4}(:,1)-zf_index_simu)*dz*1e3,10*log10(SNRs_all_fitpeak{4}),'--k','LineWidth',1.5)
   470	% DFT+ISAM
   471	errorbar(z_means,SNR_means(1,:,1),SNR_means(1,:,2)-SNR_means(1,:,1),abs(SNR_means(1,:,3)-SNR_means(1,:,1)),'ob','LineWidth',0.5,'MarkerEdgeColor','b','MarkerSize',3,'CapSize',3);
   472	% MIAA+ISAM
   473	errorbar(z_means,SNR_means(2,:,1),SNR_means(2,:,2)-SNR_means(2,:,1),abs(SNR_means(2,:,3)-SNR_means(2,:,1)),'or','LineWidth',0.5,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',2,'CapSize',3);
   474	
   475	xlabel(['z - z_f (',char(181),'m)'])
   476	ylabel('SNR (dB)')
   477	box on;
   478	xlim(zlims)
   479	ylim(ylimSNR)
   480	
   481	
   482	subplot(3,6,14:15)
   483	% plot axial resolutions
   484	hold on
   485	scatter((C_images2(:,1)-zf_index)*dz*1e3,2*sqrt(log(2))*sigmaszxy1_s2(:,1)*1e3,3,'o','MarkerEdgeColor','b')
   486	scatter((C_images2(:,1)-zf_index)*dz*1e3,2*sqrt(log(2))*sigmaszxy2_s2(:,1)*1e3,0.5,'o','MarkerEdgeColor','r','MarkerFaceColor','r')
   487	
   488	plot((Cs{3}(:,1)-zf_index_simu)*dz*1e3,2*sqrt(log(2))*abs(params2{5}(:,8))*1e3*dz,'-.k','LineWidth',1.5)
   489	plot((Cs{4}(:,1)-zf_index_simu)*dz*1e3,2*sqrt(log(2))*abs(params2{6}(:,8))*1e3*dz,':k','LineWidth',1.5)
   490	plot((Cs{3}(:,1)-zf_index_simu)*dz*1e3,2*sqrt(log(2))*sigmaszxys{3}(:,1)*1e3,'-k','LineWidth',1.5)
   491	plot((Cs{4}(:,1)-zf_index_simu)*dz*1e3,2*sqrt(log(2))*sigmaszxys{4}(:,1)*1e3,'--k','LineWidth',1.5)
   492	
   493	
   494	xlabel(['z - z_f (',char(181),'m)'])
   495	ylabel(['FWHM_z (',char(181),'m)'])
   496	box on;
   497	xlim(zlims)
   498	ylim(dzlims)
   499	leg = legend({'DFT+ISAM experiment','MIAA+ISAM experiment','DFT simulation','RFIAA simulation','DFT+ISAM simulation','MIAA+ISAM simulation'},'numColumns',3);
   500	
   501	% plot lateral resolutions
   502	subplot(3,6,16:17)
   503	hold on
   504	scatter((C_images2(:,1)-zf_index)*dz*1e3,2*sqrt(log(2))*sigmaszxy1_s2(:,2)*1e3,3,'o','MarkerEdgeColor','b')
   505	scatter((C_images2(:,1)-zf_index)*dz*1e3,2*sqrt(log(2))*sigmaszxy2_s2(:,2)*1e3,0.5,'o','MarkerEdgeColor','r','MarkerFaceColor','r')
   506	
   507	plot((Cs{3}(:,1)-zf_index_simu)*dz*1e3,2*sqrt(log(2))*sigmaszxys{3}(:,2)*1e3,'-k','LineWidth',1.5)
   508	plot((Cs{4}(:,1)-zf_index_simu)*dz*1e3,2*sqrt(log(2))*sigmaszxys{4}(:,2)*1e3,'--k','LineWidth',1.5)
   509	plot((Cs{3}(:,1)-zf_index_simu)*dz*1e3,2*sqrt(log(2))*abs(params2{5}(:,6))*1e3*dx,'-.k','LineWidth',1.5)
   510	plot((Cs{4}(:,1)-zf_index_simu)*dz*1e3,2*sqrt(log(2))*abs(params2{6}(:,6))*1e3*dx,':k','LineWidth',1.5)
   511	xlabel(['z - z_f (',char(181),'m)'])
   512	ylabel(['FWHM_x (',char(181),'m)'])
   513	box on;
   514	xlim(zlims)
   515	ylim(dxylims)
   516	
   517	
   518	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   519	% layout figure
   520	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   521	
   522	for i=8:11
   523	    fig10.Children(i).XTick = [];
   524	    fig10.Children(i).YTick = [];
   525	end
   526	for i=14:17
   527	    fig10.Children(i).XTick = [];
   528	    fig10.Children(i).YTick = [];
   529	    fig10.Children(i).Title.Position(2) = fig10.Children(i).Title.Position(2)+0.5;
   530	end
   531	
   532	figratio = fig10.Position(3)/fig10.Position(4);
   533	ratio_volumes = (length(x)*dx*1e3)/(length(z)*dz*1e3)/figratio;
   534	
   535	height_volumes = 0.22;
   536	height_Ascans = 0.160;
   537	xpos = (0:4)*(height_volumes*ratio_volumes+0.005)+0.05;
   538	width_legend1 = 0.09;
   539	height_legend = 0.2;
   540	width_lineplots = (1-xpos(5)-0.132-width_legend1)/2;
   541	xpos(5) = xpos(5)+0.06;
   542	xpos(6) = xpos(5)+width_lineplots+0.06;
   543	xpos(7) = xpos(6)+width_lineplots+0.008;
   544	% axisspace = 0.065;
   545	fig10.Children(17).YLabel.String = 'at focus';
   546	fig10.Children(17).YLabel.FontWeight = 'Bold';
   547	fig10.Children(11).YLabel.String = 'z - z_f = '+string(-(round((Cs{1}(psf_indices(2,1),1)-zf_index)*dz*1e3)))+' '+char(181)+'m';
   548	fig10.Children(11).YLabel.FontWeight = 'Bold';
   549	for i = 11:14
   550	    fig10.Children(i+3).Position = [xpos(15-i),1-height_volumes-0.07,height_volumes*ratio_volumes,height_volumes];
   551	end
   552	for i = 9:10
   553	    fig10.Children(i+3).Position = [xpos(15-i),1-height_Ascans-0.07,width_lineplots,height_Ascans];
   554	    fig10.Children(i+3).XLabel.Position(2) = fig10.Children(i+3).XLabel.Position(2)*0.7;
   555	end
   556	for i = 5:8
   557	    fig10.Children(i+3).Position = [xpos(9-i),1-2*height_volumes-0.08,height_volumes*ratio_volumes,height_volumes];
   558	end
   559	fig10.Children(11).YLabel.Position(2) = 0.5;
   560	fig10.Children(5).ItemTokenSize=[8,1];
   561	pos = [xpos(7),1-height_volumes-0.075-height_legend/2+(height_volumes-height_Ascans)/2,width_legend1,height_legend];
   562	fig10.Children(5).Position = pos;
   563	
   564	
   565	
   566	for i = 3:4
   567	    fig10.Children(i+3).Position = [xpos(9-i),1-height_volumes-0.08-height_Ascans,width_lineplots,height_Ascans];
   568	    fig10.Children(i+3).XLabel.Position(2) = fig10.Children(i+3).XLabel.Position(2)*0.7;
   569	end
   570	
   571	width_resolutionplots = 0.265;
   572	height_resolutionplots = 0.29;
   573	ypos_resolutionplots = 0.08;
   574	axes_space = (xpos(7)+width_legend1-xpos(1)-0.010-3*width_resolutionplots)/2;
   575	fig10.Children(4).Position = [xpos(1)+0.010,ypos_resolutionplots,width_resolutionplots,height_resolutionplots];
   576	fig10.Children(4).XLabel.Position(2) = fig10.Children(4).YLim(1)-0.1*(fig10.Children(4).YLim(2)-fig10.Children(4).YLim(1));
   577	fig10.Children(4).YTick = 20:20:80;
   578	fig10.Children(3).Position = [xpos(1)+0.010+axes_space-0.007+width_resolutionplots,ypos_resolutionplots,width_resolutionplots,height_resolutionplots];
   579	fig10.Children(3).YLabel.Position(1) = fig10.Children(3).YLabel.Position(1)+10;
   580	fig10.Children(3).XLabel.Position(2) = fig10.Children(3).YLim(1)-0.1*(fig10.Children(3).YLim(2)-fig10.Children(3).YLim(1));
   581	fig10.Children(2).Position = [xpos(1)+0.010,ypos_resolutionplots+height_resolutionplots+0.024,xpos(7)-xpos(1)-0.010+width_legend1,0.06];
   582	fig10.Children(1).Position = [xpos(7)+width_legend1-width_resolutionplots,ypos_resolutionplots,width_resolutionplots,height_resolutionplots];
   583	fig10.Children(1).YLabel.Position(1) = fig10.Children(1).YLabel.Position(1)+15;
   584	fig10.Children(1).XLabel.Position(2) = fig10.Children(1).YLim(1)-0.1*(fig10.Children(1).YLim(2)-fig10.Children(1).YLim(1));
   585	fig10.Children(1).YTick = 0.7:0.3:1.95;
   586	%
   587	ypos = [1-0.030,1-0.045-height_volumes-0.000,1-2*height_volumes+height_Ascans-0.05,ypos_resolutionplots+height_resolutionplots+0.0260];
   588	fig.a(1) = annotation('textbox','string','(a)','FontWeight','bold','color','k','Position',[0.000,ypos(1),0,0],'EdgeColor','none');
   589	fig.a(2) = annotation('textbox','string','(b)','FontWeight','bold','color','k','Position',[xpos(5)-0.06,ypos(1),0,0],'EdgeColor','none');
   590	fig.a(3) = annotation('textbox','string','(c)','FontWeight','bold','color','k','Position',[xpos(6)-0.06,ypos(1),0,0],'EdgeColor','none');
   591	
   592	fig.a(4) = annotation('textbox','string','(d)','FontWeight','bold','color','k','Position',[0.000,ypos(2),0,0],'EdgeColor','none');
   593	fig.a(5) = annotation('textbox','string','(e)','FontWeight','bold','color','k','Position',[xpos(5)-0.06,ypos(2),0,0],'EdgeColor','none');
   594	fig.a(6) = annotation('textbox','string','(f)','FontWeight','bold','color','k','Position',[xpos(6)-0.06,ypos(2),0,0],'EdgeColor','none');
   595	
   596	fig.a(7) = annotation('textbox','string','(g)','FontWeight','bold','color','k','Position',[0.000-0.01,ypos(4),0,0],'EdgeColor','none');
   597	fig.a(8) = annotation('textbox','string','(h)','FontWeight','bold','color','k','Position',[0.000+width_resolutionplots+axes_space-0.007,ypos(4),0,0],'EdgeColor','none');
   598	fig.a(9) = annotation('textbox','string','(i)','FontWeight','bold','color','k','Position',[xpos(7)+width_legend1-width_resolutionplots-xpos(1)-0.025,ypos(4),0,0],'EdgeColor','none');
   599	
   600	%%
   601	fig.name = 'figure4.pdf';
   602	fig.pos = get(fig10,'Position');
   603	set(fig10,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[fig.pos(3), fig.pos(4)])
   604	print(fig10, '-dpdf', fig.name);
   605	print(fig10, '-dpng','-r500', [fig.name(1:end-4),'.png']);
```
