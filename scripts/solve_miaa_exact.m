function out=solve_miaa_exact(y,Fg,Ffull,input_in_full,K,q,eta_weight)
  [a,PE,~]=fiaa_oct_c1(y,K,q,eta_weight);
  p=abs(a).^2;
  eta=max(real(PE(q+1)),1e-12*mean(abs(y).^2));
  R=(Fg.*p)*Fg'+eta*eye(length(y));
  alpha=Fg'*(R\y);
  yf=Ffull*(p.*alpha);
  yf(input_in_full)=y;
  nf=(-size(Ffull,1)/2:size(Ffull,1)/2-1)';
  spec=zeros(K,1);
  for j=1:length(nf)
    spec(mod(nf(j),K)+1)=yf(j);
  end
  out=ifft(spec)*K;
end
