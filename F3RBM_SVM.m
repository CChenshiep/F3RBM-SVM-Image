load MNIST_Handwritten_data
maxepoch=20; %In the Science paper we use maxepoch=50, but it works just fine. 
numhid=500; 
[numcases numdims numbatches]=size(batchdata);
restart=1;
epsilonw      =0.1;   % Learning rate for weights 
epsilonvb     =0.1;   % Learning rate for biases of visible units 
epsilonhb     =0.1;   % Learning rate for biases of hidden units 
 weightcost  =  0.0002; 
initialmomentum  = 0.5;
finalmomentum    = 0.9;
if restart ==1
  restart=0;
  epoch=1;
vishid1=0.01*randn(2*numdims, numhid);
% vimin=min(min(vishid1));vimax=max(max(vishid1));
a=unique(vishid1);t=size(a,1);tl=t/2;
al=a(1:tl);ar=a(tl+1:t);
% (randperm(numel(al)))
wl=reshape(al, numdims,numhid);
wr=reshape(ar,numdims,numhid );
ho=0.01*randn(1, 2*numhid);a=unique(ho);t=size(a,2);tl=t/2;
hl=a(1:tl);hr=a(tl+1:t);
vo=0.01*randn(1, 2*numdims);a=unique(vo);t=size(a,2);tl=t/2;
vl=a(1:tl);vr=a(tl+1:t);
vishidincl  = zeros(numdims,numhid);
vishidincr  = zeros(numdims,numhid);
hidbiasincl = zeros(1,numhid);
hidbiasincr = zeros(1,numhid);
visbiasincl = zeros(1,numdims);
visbiasincr = zeros(1,numdims);
% batchposhidprobs=zeros(numcases,numhid);
end
for epoch = epoch:maxepoch
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
    for batch = 1:numbatches,
%    fprintf(1,'epoch %d batch %d\r',epoch); 

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data1 = batchdata(:,:,batch);
poshidprobsl = 1./(1 + exp(-data1*wl - repmat(hl,numcases,1)));
poshidprobsr = 1./(1 + exp(-data1*wr - repmat(hr,numcases,1)));
batchposhidprobs=(poshidprobsl+poshidprobsr)/2;
[m,n]=size (batchposhidprobs); 
 po= batchposhidprobs > rand(m,n);
X=po*2-1;Y=X'*X;

p=[];t=1;

for i=1:n-1
    for j=i+1:n
        if Y(i,j)>=0.97*m
            p(1,t)=i;
            p(2,t)=j;
            t=t+1;
        end
    end
end
if (~isempty(p))
q=unique(p(2,:));
% h0(:,q)=[];
 A1=unique(p(1,:));
[a1,b1]=size(A1);
A2=hist(p(1,:),A1);
A3=cumsum(A2);
t=1;
for i=1:b1
    wl(:,A1(i))=sum(wl(:,p(2,t:A3(i)))')'+wl(:,A1(i));
   t=A3(i)+1;
end
for i=1:b1
    wr(:,A1(i))=sum(wr(:,p(2,t:A3(i)))')'+wr(:,A1(i));
   t=A3(i)+1;
end
for i=1:b1
    hl(A1(i))=sum(hl(p(2,t:A3(i)))')'+hl(A1(i));
   t=A3(i)+1;
end
for i=1:b1
    hr(A1(i))=sum(hr(p(2,t:A3(i)))')'+hr(A1(i));
   t=A3(i)+1;
end
wl(:,q)=[];
wr(:,q)=[];
vishidincl(:,q)=[];
vishidincr(:,q)=[];

hidbiasincl(:,q) = [];
hidbiasincr(:,q) = [];
hl(:,q)=[];
hr(:,q)=[];
poshidprobsl(:,q) = [];
poshidprobsr(:,q) = [];
end 
[mm,nn]=size(poshidprobsl);
poshidstatesl = poshidprobsl > rand(mm,nn);
poshidstatesr = poshidprobsr > rand(mm,nn);
%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  negdatal = 1./(1 + exp(-poshidstatesl*wl' - repmat(vl,numcases,1)));
  negdatar = 1./(1 + exp(-poshidstatesr*wr' - repmat(vr,numcases,1)));
  neghidprobsl = 1./(1 + exp(-negdatal*wl - repmat(hl,numcases,1))); 
  neghidprobsr = 1./(1 + exp(-negdatar*wr - repmat(hr,numcases,1)));

x11=(negdatal+negdatar)/2;

   err= ((sum(sum( (data1-x11).^2 )))/numcases);

  errsum = err + errsum;

 if epoch>5
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
 end;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
vishidincl = (momentum*vishidincl + ...
            epsilonw*( (data1'*poshidprobsl-negdatal'* neghidprobsl)/(numcases) - weightcost*wl));
vishidincr =( momentum*vishidincr + ...
            epsilonw*( (data1'*poshidprobsr-negdatar'* neghidprobsr)/(numcases) - weightcost*wr));
visbiasincl = (momentum*visbiasincl + (epsilonvb/numcases)*(sum(data1)-sum(negdatal)));
visbiasincr = (momentum*visbiasincr + (epsilonvb/numcases)*(sum(data1)-sum(negdatar)));
hidbiasincl =  (momentum*hidbiasincl + (epsilonvb/numcases)*(sum(poshidprobsl)-sum(neghidprobsl)));
hidbiasincr =(momentum*hidbiasincr + (epsilonvb/numcases)*(sum(poshidprobsr)-sum(neghidprobsr)));
% vishidinc=(vishidincl+vishidincr)/2;visbiasinc=(visbiasincl+visbiasincr)/2;hidbiasinc=(hidbiasincl +hidbiasincr)/2;
wl= wl+ vishidincl;wr=wr+vishidincr;
vl = vl + visbiasincl;vr = vr + visbiasincr;
hl = hl + hidbiasincl; hr = hr + hidbiasincr;
    end 

errsum=sqrt(errsum);
fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 


end


output=[];
 for ii=1:100
  output = [output data1(ii,1:end)' x11(ii,:)'];
 end
   if epoch==1 
   close all 
%    figure('Position',[100,600,1000,200]);
   else 
   figure(4)
   end 
   mnistdisp(output);
   drawnow;
   
 %%%%%%%%SVM%%%%%%%  
[numcases1,numdims1]=size(traindata);
[numcases2,numdims2]=size(testdata);
poshidprobsl = 1./(1 + exp(-traindata*wl - repmat(hl,numcases1,1)));
poshidprobsr = 1./(1 + exp(-traindata*wr - repmat(hr,numcases1,1)));
tposhidprobsl = 1./(1 + exp(-testdata*wl - repmat(hl,numcases2,1)));
tposhidprobsr = 1./(1 + exp(-testdata*wr - repmat(hr,numcases2,1)));

train_set=poshidprobsl+poshidprobsr;
test_set=tposhidprobsl+tposhidprobsr;
[mtrain,ntrain] = size(train_set);
[mtest,ntest] = size(test_set);
test_dataset = [train_set;test_set];
[dataset_scale,ps] = mapminmax(test_dataset',0,1);%y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin

model = svmtrain(train_set_labels, train_set,'-c 10,-g 0.01 ');
[predict_label, accuracy, prob_estimates] = svmpredict(test_set_labels, test_set, model);  

