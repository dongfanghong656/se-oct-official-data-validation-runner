# oct_iaa_c1.m — checksum-traced exact wrapper source

- Source: Zenodo record 7870795
- MD5: `d189d45be2e5f5ac3b797fb80e7b84e3`
- Purpose: audit chunk partitioning, traversal direction, and output assembly.

```matlab
     1	function [S,Smap] =oct_iaa_c1(X,q_i,K,eta,isel,q_rec,L)
     2	% this fuction applies the non-recursive or recursive version of FIAA,
     3	% dependent on the value of isel {1,2}
     4	% X => the uniformly reshaped interference spectrum
     5	% q => the number of iterations
     6	% K => the length of the reconstruction grid in depth direction
     7	% eta => a weighting of the noise matrix in R, default is 1
     8	% isel {1,2} => chooses between nonrecursive{1} and recursive{2} reconstrution
     9	% qrec => the number of iteration of recursive s
    10	% L => the number of sections for the parallel processing, ideally chosen
    11	% equal to the number of CPU cores or a multiple of that.
    12	% S <= the IAA reconstrution
    13	
    14	 meth=[...
    15	 'FIAA  ';
    16	 'RFIAA '];
    17	
    18	[iM,iN]=size(X);
    19	if(isel==1)  %       Fast IAA - Individual Ascan processing
    20	%     disp(['Method ' meth(isel,:)])
    21	    S=zeros(K,iN); Smap=S;
    22	    parfor i=1:iN
    23	      [S(:,i),~, Smap(:,i)]= fiaa_oct_c1(X(:,i),K,q_i,eta);
    24	    end
    25	      S=[S(1,:); S(end:-1:2,:)];  
    26	      Smap=[Smap(1,:); Smap(end:-1:2,:)];  
    27	
    28	elseif(isel==2) %       Fast IAA - Warm startup Ascan processing
    29	%     disp(['Method ' meth(isel,:)])
    30	    NL=floor(iN/L);
    31	    iNL=L*NL;
    32	    S=zeros(K,NL,L);
    33	    Smap=S;
    34	    bX=X(:,1:iNL);
    35	    bX=reshape(bX,iM,NL,L);
    36	    parfor ip=1:L
    37	    [S(:,:,ip),Smap(:,:,ip)]=rfiaa_oct_c1(squeeze(bX(:,:,ip)),K,q_i,eta,q_rec);  
    38	    end
    39	    S=reshape(S,K,iNL);
    40	    Smap=reshape(Smap,K,iNL);
    41	    S=[S(1,:); S(end:-1:2,:)];  
    42	    Smap=[Smap(1,:); Smap(end:-1:2,:)];  
    43	
    44	else  
    45	   
    46	    disp('Wrong choice')
    47	end
    48	
    49	 
```
