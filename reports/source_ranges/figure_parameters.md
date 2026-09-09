# Verified plotting parameter audit

## plot_figure3.m
```matlab
    10	%define the indices of the plotted frames and the x,y and z limits
    11	Yindex = 248; 
    12	Zindex1 = 566; %PSF planes
    13	Zindex2 = 461;
    14	zlims = [0.25,0.65]; 
    15	xlims_bscan = [0,0.225]; 
    16	xlims_enface1 = [0.121,0.221]; 
    17	ylims_enface1 = [0.05,0.15]; 
    18	xlims_enface2 = [0.1,0.20]; 
    19	ylims_enface2 = [0.05,0.15]; 
    22	methods = {'DFT','RFIAA','DFT + ISAM','MIAA + ISAM'};
    28	    imagesc(Xroi,Zroi,squeeze(image(:,:,Yindex)))
    32	    ylim(zlims)
    33	    xlim(xlims_bscan)
    35	    plot(xlims_enface1,[1,1]*z_isam(Zindex1),'-b','LineWidth',1.5)
    36	    plot(xlims_enface2,[1,1]*z_isam(Zindex2),'-r','LineWidth',1.5)
    37	    plot(xlims_bscan(1)+(xlims_bscan(2)-xlims_bscan(1))/15+[0,scalebar_length],(zlims(2)-(zlims(2)-zlims(1))/40)*[1,1],'-w','LineWidth',3)
    39	        text(xlims_bscan(1)+(xlims_bscan(2)-xlims_bscan(1))/15+scalebar_length/2,(zlims(2)-3*(zlims(2)-zlims(1))/40),string(round(scalebar_length*1e3))+' '+char(181)+'m','Color','w','HorizontalAlignment','Center')
    41	    title(methods{i})
    46	    imagesc(Xroi,Xroi,squeeze(image(Zindex1,:,:))')
    49	    xlim(xlims_enface1)
    50	    ylim(ylims_enface1)
    52	    plot(xlims_enface1(1)+(xlims_enface1(2)-xlims_enface1(1))/12+[0,scalebar_length2],(ylims_enface1(2)-(ylims_enface1(2)-ylims_enface1(1))/20)*[1,1],'-w','LineWidth',3)
    54	        text(xlims_enface1(1)+(xlims_enface1(2)-xlims_enface1(1))/12+scalebar_length2/2,(ylims_enface1(2)-3*(ylims_enface1(2)-ylims_enface1(1))/20),string(round(scalebar_length2*1e3))+' '+char(181)+'m','Color','w','HorizontalAlignment','Center')
    62	    imagesc(Xroi,Xroi,squeeze(image(Zindex2,:,:))')
    65	    xlim(xlims_enface2)
    66	    ylim(ylims_enface2)
    68	    plot(xlims_enface2(1)+(xlims_enface2(2)-xlims_enface2(1))/12+[0,scalebar_length2],(ylims_enface2(2)-(ylims_enface2(2)-ylims_enface2(1))/20)*[1,1],'-w','LineWidth',3)
    70	        text(xlims_enface2(1)+(xlims_enface2(2)-xlims_enface2(1))/12+scalebar_length2/2,(ylims_enface2(2)-3*(ylims_enface2(2)-ylims_enface2(1))/20),string(round(scalebar_length2*1e3))+' '+char(181)+'m','Color','w','HorizontalAlignment','Center')
    79	ratio_Bscan = (xlims_bscan(2)-xlims_bscan(1))/(zlims(2)-zlims(1))*figratio;
    80	ratio_enface1 = (xlims_enface1(2)-xlims_enface1(1))/(ylims_enface1(2)-ylims_enface1(1))*figratio;
    81	ratio_enface2 = (xlims_enface2(2)-xlims_enface2(1))/(ylims_enface2(2)-ylims_enface2(1))*figratio;
    88	fig.ratio(1) = (xlims_bscan(2)-xlims_bscan(1))/(zlims(2)-zlims(1))/figratio;
    89	fig.ratio(2) = (xlims_enface1(2)-xlims_enface1(1))/(ylims_enface1(2)-ylims_enface1(1))/figratio;
    90	fig.ratio(3) = (xlims_enface2(2)-xlims_enface2(1))/(ylims_enface2(2)-ylims_enface2(1))/figratio;
   106	    fig3.Children(ch_no1).YAxis.LineWidth = 1.5;
   107	    fig3.Children(ch_no1).XAxis.LineWidth = 1.5;
   111	    fig3.Children(ch_no2).YAxis.LineWidth = 1.5;
   112	    fig3.Children(ch_no2).XAxis.LineWidth = 1.5;
   134	fig3.Children(14).YLabel.String = 'in focus';
   136	fig3.Children(13).YLabel.String = 'out of focus';
   151	t(1) = text(0.1245,0.108,'\leftarrow','Rotation',-30,'Color','w','parent',fig3.Children(2),'FontSize',14,'FontWeight','Bold');
   152	t(2) = text(0.1245,0.108,'\leftarrow','Rotation',-30,'Color','w','parent',fig3.Children(7),'FontSize',14,'FontWeight','Bold');
   153	t(3) = text(0.1985,0.0975,'\rightarrow','Rotation',-30,'Color','w','parent',fig3.Children(4),'FontSize',14,'FontWeight','Bold');
   154	t(4) = text(0.1985,0.0975,'\rightarrow','Rotation',-30,'Color','w','parent',fig3.Children(8),'FontSize',14,'FontWeight','Bold');
   155	t(5) = text(0.1985,0.0975,'\rightarrow','Rotation',-30,'Color','w','parent',fig3.Children(11),'FontSize',14,'FontWeight','Bold');
   156	t(6) = text(0.1985,0.0975,'\rightarrow','Rotation',-30,'Color','w','parent',fig3.Children(14),'FontSize',14,'FontWeight','Bold');
   158	t(7) = text(xlims_bscan(1)+0.005,zlims(1)+0.01,'\rightarrowx','Color','w','parent',fig3.Children(15),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','left');
   159	t(8) = text(xlims_bscan(1)+0.01,zlims(1)+0.004,'\downarrow','Color','w','parent',fig3.Children(15),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   160	t(9) = text(xlims_bscan(1)+0.01,zlims(1)+0.03,'z','Color','w','parent',fig3.Children(15),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   162	t(10) = text(xlims_enface1(1)+0.0035,ylims_enface1(1)+0.004,'\rightarrowx','Color','w','parent',fig3.Children(14),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','left');
   163	t(11) = text(xlims_enface1(1)+0.005,ylims_enface1(1)+0.002,'\downarrow','Color','w','parent',fig3.Children(14),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   164	t(12) = text(xlims_enface1(1)+0.005,ylims_enface1(1)+0.015,'y','Color','w','parent',fig3.Children(14),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   166	t(13) = text(xlims_enface2(1)+0.0035,ylims_enface2(1)+0.004,'\rightarrowx','Color','w','parent',fig3.Children(13),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','left');
   167	t(14) = text(xlims_enface2(1)+0.005,ylims_enface2(1)+0.002,'\downarrow','Color','w','parent',fig3.Children(13),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   168	t(15) = text(xlims_enface2(1)+0.005,ylims_enface2(1)+0.015,'y','Color','w','parent',fig3.Children(13),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
```

## plot_figure5.m
```matlab
    11	%define indices of plotted planes and x and y limits for the figures
    12	Yindex = 256;
    13	ypos_bscan = Yindex/Nx*sizeX;
    14	Zindex1 = 720; 
    15	Zindex2 = 637;
    16	zlims = [0.465,0.755]; 
    17	xlims_bscan = [0,0.225]; 
    18	xlims_enface1 = [0,0.225]; 
    19	ylims_enface1 = [0,0.225]; 
    20	xlims_enface1_2 = [0.082,0.144]; 
    21	ylims_enface1_2 = [0.1370,0.1990];
    22	ylims_enface2 = [0.047,0.165]; 
    23	xlims_enface2 = [0.023,0.141]; 
    24	zlims_bscan2 = [0.485,0.545]; 
    25	xlims_bscan2 = [0.157,0.217]; 
    26	Ascan_zlim = [0.52,0.55];
    27	[~,Ascan_zind] = min(abs([z_isam;z_isam]-Ascan_zlim'),[],2);
    28	Ascan_xind = 163; 
    29	Ascan_xpos = Ascan_xind/Nx*sizeX;
    36	methods = {'DFT','RFIAA','DFT + ISAM','MIAA + ISAM'};
    37	methods_leg = {'DFT','RFIAA',['DFT+',newline,' ISAM'],['MIAA+',newline,' ISAM']};
    41	dBlims = [60,60,60,70]; %MIAA+ISAM has higher dB limits due to increased sparsity
    44	    %plot A-line data with double-peak gaussian fit as line (subfig (e))
    47	    AscandB = image(Ascan_zind(1):Ascan_zind(2),Ascan_xind,Yindex);
    48	    AscandB = AscandB-max(AscandB);
    49	    Ascan_int = 10.^(AscandB/10);
    50	    z_Ascan = z_isam(Ascan_zind(1):Ascan_zind(2));
    51	    z_Ascan_fitplot = linspace(z_Ascan(1),z_Ascan(end),1024);
    53	    lmaxi = find(islocalmax(Ascan_int));
    55	        list = [lmaxi, Ascan_int(lmaxi)];
    61	    param0 = [Ascan_int(lmaxi(1)),z_Ascan(lmaxi(1)),0.002,Ascan_int(lmaxi(2)),z_Ascan(lmaxi(2)),0.002];
    62	    params(i,:) = lsqcurvefit(Fg,param0,z_Ascan,Ascan_int');
    63	    plot(z_Ascan,Ascan_int,'o','Color',colors(i),'LineWidth',1,'DisplayName',methods_leg{i},'MarkerSize',1.5,'MarkerFaceColor',colors(i),'HandleVisibility','off')
    64	    plot(z_Ascan_fitplot,Fg(params(i,:),z_Ascan_fitplot),'-','Color',colors(i),'LineWidth',0.5,'DisplayName',methods_leg{i},'HandleVisibility','off')
    65	    plot([-1,1],[-1,-1],'-o','Color',colors(i),'LineWidth',0.5,'DisplayName',methods_leg{i},'MarkerSize',1.5,'MarkerFaceColor',colors(i))
    69	        xlim(Ascan_zlim)
    70	        ylim([0,1.1]);
    78	        imagesc(Xroi,Zroi,squeeze(image(:,:,Yindex)))
    79	        caxis([-dBlims(i),0])
    82	        ylim(zlims)
    83	        xlim(xlims_bscan)
    85	        plot([1,1]*Ascan_xpos,Ascan_zlim,'-w','LineWidth',1)
    86	        plot(xlims_enface1,[1,1]*z_isam(Zindex1),'--w','LineWidth',1.0)
    87	        plot(xlims_enface2,[1,1]*z_isam(Zindex2),'--g','LineWidth',1.0)
    88	        plot(xlims_bscan2*xbox_transform,zlims_bscan2*ybox_transform,'-y','LineWidth',1.0)
    89	        plot(xlims_bscan(1)+(xlims_bscan(2)-xlims_bscan(1))/20+[0,scalebar_length],(zlims(2)-(zlims(2)-zlims(1))/40)*[1,1],'-w','LineWidth',3)
    91	            text(xlims_bscan(1)+(xlims_bscan(2)-xlims_bscan(1))/20+scalebar_length/2,(zlims(2)-3*(zlims(2)-zlims(1))/40),string(round(scalebar_length*1e3))+' '+char(181)+'m','Color','w','HorizontalAlignment','Center')
    93	        title(methods{i})
    96	        imagesc(Xroi,Xroi,flip(squeeze(image(Zindex1,:,:)),2)')
    97	        caxis([-dBlims(i),0])
    99	        xlim(xlims_enface1)
   100	        ylim(ylims_enface1)
   101	        title(methods{i})
   103	        plot(xlims_enface1_2*xbox_transform,ylims_enface1_2*ybox_transform,'-b','LineWidth',1.0)
   104	        plot([xlims_enface1_2(1),xlims_enface1(1)],[ylims_enface1_2(2),ylims_enface1(2)],'-b','LineWidth',1.0)
   105	        plot([xlims_enface1_2(2),xlims_enface1(2)],[ylims_enface1_2(2),ylims_enface1(2)],'-b','LineWidth',1.0)
   107	        plot(ylims_enface1,ypos_bscan*[1,1],'--w','LineWidth',1)    
   108	        plot(xlims_enface1(1)+(xlims_enface1(2)-xlims_enface1(1))/20+[0,scalebar_length],(ylims_enface1(2)-(ylims_enface1(2)-ylims_enface1(1))/20)*[1,1],'-w','LineWidth',3)
   110	            text(xlims_enface1(1)+(xlims_enface1(2)-xlims_enface1(1))/20+scalebar_length/2,(ylims_enface1(2)-2.5*(ylims_enface1(2)-ylims_enface1(1))/20),string(round(scalebar_length*1e3))+' '+char(181)+'m','Color','w','HorizontalAlignment','Center')
   112	        % % % zoom in enface image (c)
   114	        imagesc(Xroi,Xroi,flip(squeeze(image(Zindex1,:,:)),2)')
   115	        caxis([-dBlims(i),0])
   117	        xlim(xlims_enface1_2)
   118	        ylim(ylims_enface1_2)
   120	        plot(xlims_enface1_2(1)+(xlims_enface1_2(2)-xlims_enface1_2(1))/20+[0,scalebar_length_2],(ylims_enface1_2(2)-(ylims_enface1_2(2)-ylims_enface1_2(1))/20)*[1,1],'-w','LineWidth',3)
   122	            text(xlims_enface1_2(1)+(xlims_enface1_2(2)-xlims_enface1_2(1))/20+scalebar_length_2/2,(ylims_enface1_2(2)-2.5*(ylims_enface1_2(2)-ylims_enface1_2(1))/20),string(round(scalebar_length_2*1e3))+' '+char(181)+'m','Color','w','HorizontalAlignment','Center')
   127	    imagesc(Xroi,Xroi,flip(squeeze(image(Zindex2,:,:)),2)')
   128	    caxis([-dBlims(i),0])
   130	    xlim(xlims_enface2)
   131	    ylim(ylims_enface2)
   133	    plot(xlims_enface2(1)+(xlims_enface2(2)-xlims_enface2(1))/20+[0,scalebar_length],(ylims_enface2(2)-(ylims_enface2(2)-ylims_enface2(1))/20)*[1,1],'-w','LineWidth',3)
   135	        text(xlims_enface2(1)+(xlims_enface2(2)-xlims_enface2(1))/20+scalebar_length/2,(ylims_enface2(2)-3.1*(ylims_enface2(2)-ylims_enface2(1))/20),string(round(scalebar_length*1e3))+' '+char(181)+'m','Color','w','HorizontalAlignment','Center')
   138	        plot(xlims_enface2,ypos_bscan*[1,1],'--g','LineWidth',1)
   140	    title(methods{i})
   143	    imagesc(Xroi,Zroi,squeeze(image(:,:,Yindex)))
   144	    caxis([-dBlims(i),0])
   146	    ylim(zlims_bscan2)
   147	    xlim(xlims_bscan2)
   149	    plot(xlims_bscan2(1)+(xlims_bscan2(2)-xlims_bscan2(1))/20+[0,scalebar_length_2],(zlims_bscan2(2)-(zlims_bscan2(2)-zlims_bscan2(1))/20)*[1,1],'-w','LineWidth',3)
   151	        text(xlims_bscan2(1)+(xlims_bscan2(2)-xlims_bscan2(1))/20+scalebar_length_2/2,(zlims_bscan2(2)-1.7*(zlims_bscan2(2)-zlims_bscan2(1))/10),string(round(scalebar_length_2*1e3))+' '+char(181)+'m','Color','w','HorizontalAlignment','Center')
   153	    title(methods{i})
   158	ratio_Bscan = (xlims_bscan(2)-xlims_bscan(1))/(zlims(2)-zlims(1))*figratio;
   159	ratio_enface1 = (xlims_enface1(2)-xlims_enface1(1))/(ylims_enface1(2)-ylims_enface1(1))*figratio;
   160	ratio_enface2 = (xlims_enface2(2)-xlims_enface2(1))/(ylims_enface2(2)-ylims_enface2(1))*figratio;
   167	fig.ratio(1) = (xlims_bscan(2)-xlims_bscan(1))/(zlims(2)-zlims(1))/figratio;
   168	fig.ratio(2) = (xlims_enface1(2)-xlims_enface1(1))/(ylims_enface1(2)-ylims_enface1(1))/figratio;
   169	fig.ratio(3) = (xlims_enface1_2(2)-xlims_enface1_2(1))/(ylims_enface1_2(2)-ylims_enface1_2(1))/figratio;
   170	fig.ratio(4) = (xlims_enface2(2)-xlims_enface2(1))/(ylims_enface2(2)-ylims_enface2(1))/figratio;
   188	    fig3.Children(ch_no).YAxis.LineWidth = 1.5;
   189	    fig3.Children(ch_no).XAxis.LineWidth = 1.5;
   193	    fig3.Children(ch_no).YAxis.LineWidth = 1.5;
   194	    fig3.Children(ch_no).XAxis.LineWidth = 1.5;
   198	    fig3.Children(ch_no).YAxis.LineWidth = 1.5;
   199	    fig3.Children(ch_no).XAxis.LineWidth = 1.5;
   211	    fig3.Children(ch_no).YAxis.LineWidth = 1.5;
   212	    fig3.Children(ch_no).XAxis.LineWidth = 1.5;
   215	    fig3.Children(ch_no).Title.Position(2) = (fig3.Children(ch_no).YLim(2)-fig3.Children(ch_no).YLim(1))*0.01+fig3.Children(ch_no).YLim(1);
   225	    fig3.Children(ch_no).YAxis.LineWidth = 1.5;
   226	    fig3.Children(ch_no).XAxis.LineWidth = 1.5;
   229	    fig3.Children(ch_no).Title.Position(2) = (fig3.Children(ch_no).YLim(2)-fig3.Children(ch_no).YLim(1))*0.01+fig3.Children(ch_no).YLim(1);
   235	fig.Ascan_width = 0.215;
   236	fig.Ascan_height = 0.15;
   239	pos(3) = fig.Ascan_width;
   240	pos(4) = fig.Ascan_height;
   242	pos(1) = pos(1)+fig.Ascan_width+0.005;
   278	pos(2) = fig.ypos(3)+0.04+fig.Ascan_height+0.02;
   282	t(1) = text(0.056,0.111,'\rightarrow','Rotation',45,'Color','w','parent',fig3.Children(7),'FontSize',14,'FontWeight','Bold');
   283	t(2) = text(0.047,0.056,'\rightarrow','Rotation',-45,'Color','g','parent',fig3.Children(7),'FontSize',14,'FontWeight','Bold');
   284	t(3) = text(0.0770,0.058,'\rightarrow','Rotation',-60,'Color','g','parent',fig3.Children(7),'FontSize',14,'FontWeight','Bold');
   285	t(4) = text(0.081,0.159,'\rightarrow','Rotation',45,'Color','r','parent',fig3.Children(7),'FontSize',14,'FontWeight','Bold');
   287	% axis indication
   288	t(7) = text(xlims_bscan(1)+0.004,zlims(1)+0.004,'\rightarrowx','Color','w','parent',fig3.Children(10),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','left');
   289	t(8) = text(xlims_bscan(1)+0.007,zlims(1)+0.000,'\downarrow','Color','w','parent',fig3.Children(10),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   290	t(9) = text(xlims_bscan(1)+0.007,zlims(1)+0.019,'z','Color','w','parent',fig3.Children(10),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   292	t(10) = text(xlims_enface1(1)+0.005,ylims_enface1(1)+0.005,'\rightarrowx','Color','w','parent',fig3.Children(9),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','left');
   293	t(11) = text(xlims_enface1(1)+0.008,ylims_enface1(1)+0.0015,'\downarrow','Color','w','parent',fig3.Children(9),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   294	t(12) = text(xlims_enface1(1)+0.008,ylims_enface1(1)+0.020,'y','Color','w','parent',fig3.Children(9),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   296	rat = (xlims_enface1_2(2)-xlims_enface1_2(1))/(xlims_enface1(2)-xlims_enface1(1));
   297	t(13) = text(xlims_enface1_2(1)+0.005*rat,ylims_enface1_2(1)+0.005*rat,'\rightarrowx','Color','w','parent',fig3.Children(8),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','left');
   298	t(14) = text(xlims_enface1_2(1)+0.008*rat,ylims_enface1_2(1)+0.0015*rat,'\downarrow','Color','w','parent',fig3.Children(8),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   299	t(15) = text(xlims_enface1_2(1)+0.008*rat,ylims_enface1_2(1)+0.020*rat,'y','Color','w','parent',fig3.Children(8),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   301	t(16) = text(xlims_bscan2(1)+0.004,zlims_bscan2(1)+0.003,'\rightarrowx','Color','w','parent',fig3.Children(13),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','left');
   302	t(17) = text(xlims_bscan2(1)+0.005,zlims_bscan2(1)+0.001,'\downarrow','Color','w','parent',fig3.Children(13),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   303	t(18) = text(xlims_bscan2(1)+0.005,zlims_bscan2(1)+0.009,'z','Color','w','parent',fig3.Children(13),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   306	t(19) = text(xlims_enface2(1)+0.006,ylims_enface2(1)+0.005,'\rightarrowx','Color','w','parent',fig3.Children(14),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','left');
   307	t(20) = text(xlims_enface2(1)+0.008,ylims_enface2(1)+0.0015,'\downarrow','Color','w','parent',fig3.Children(14),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
   308	t(21) = text(xlims_enface2(1)+0.008,ylims_enface2(1)+0.020,'y','Color','w','parent',fig3.Children(14),'FontSize',12,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','top');
```
