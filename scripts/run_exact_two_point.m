% Exact deposited FIAA plus paper-equation MIAA on registered two-point tests.
pkg load signal;
code_dir=getenv('UPSTREAM_CODE_DIR');out_file=getenv('OUTPUT_MAT');repo_root=getenv('GITHUB_WORKSPACE');
if isempty(code_dir)||isempty(out_file),error('UPSTREAM_CODE_DIR and OUTPUT_MAT required');end
addpath(code_dir);if ~isempty(repo_root),addpath(fullfile(repo_root,'scripts'));else,addpath('scripts');end
K=256;Ng=32;Nfull=128;q=10;eta_weight=1.0;
ng_center=(-Ng/2:Ng/2-1)';nf=(-Nfull/2:Nfull/2-1)';zgrid=(0:K-1);
Fg=exp(-1i*2*pi*ng_center*zgrid/K);Ffull=exp(-1i*2*pi*nf*zgrid/K);
input_in_full=zeros(Ng,1);for i=1:Ng,input_in_full(i)=find(nf==ng_center(i));end
seps=[2,3,4,5,6,8,10,12];phases=[0,pi/2,pi];ratios=[1,0.5];snrs=[20,40];seeds=0:3;
Ncase=length(seps)*length(phases)*length(ratios)*length(snrs)*length(seeds);
meta=zeros(Ncase,5);prof_dft=complex(zeros(K,Ncase));prof_iaa=complex(zeros(K,Ncase));prof_miaa=complex(zeros(K,Ncase));
z1=96;ci=0;
for sep=seps
 for ph=phases
  for ratio=ratios
   for snr=snrs
    for seed=seeds
     ci=ci+1;randn('seed',100000+seed+100*sep+round(ph*10)+round(ratio*10));
     y=exp(-1i*2*pi*ng_center*z1/K)+ratio*exp(1i*ph)*exp(-1i*2*pi*ng_center*(z1+sep)/K);
     pow=mean(abs(y).^2);nv=pow/(10^(snr/10));yn=y+sqrt(nv/2)*(randn(Ng,1)+1i*randn(Ng,1));
     [a,~,~]=fiaa_oct_c1(yn,K,q,eta_weight);
     spec0=zeros(K,1);for j=1:Ng,spec0(mod(ng_center(j),K)+1)=yn(j);end
     prof_dft(:,ci)=ifft(spec0)*K;prof_iaa(:,ci)=a;
     prof_miaa(:,ci)=solve_miaa_exact(yn,Fg,Ffull,input_in_full,K,q,eta_weight);
     meta(ci,:)=[sep,ph,ratio,snr,seed];
    end
   end
  end
 fprintf('sep %d complete\n',sep);
end
add_meta=[];add_out_pair=[];add_out_sum=[];
for sep=[2,4,8,12]
 for ratio=[1,0.5]
  for ph=[0,pi/2,pi]
   y1=exp(-1i*2*pi*ng_center*z1/K);y2=ratio*exp(1i*ph)*exp(-1i*2*pi*ng_center*(z1+sep)/K);
   o1=solve_miaa_exact(y1,Fg,Ffull,input_in_full,K,q,eta_weight);
   o2=solve_miaa_exact(y2,Fg,Ffull,input_in_full,K,q,eta_weight);
   op=solve_miaa_exact(y1+y2,Fg,Ffull,input_in_full,K,q,eta_weight);
   add_meta=[add_meta;sep,ratio,ph];add_out_pair=[add_out_pair,op];add_out_sum=[add_out_sum,o1+o2];
  end
 end
end
save('-mat7-binary',out_file,'K','Ng','Nfull','q','eta_weight','seps','phases','ratios','snrs','seeds','z1','meta','prof_dft','prof_iaa','prof_miaa','add_meta','add_out_pair','add_out_sum');
fprintf('EXACT_TWO_POINT_OK cases=%d output=%s\n',Ncase,out_file);
