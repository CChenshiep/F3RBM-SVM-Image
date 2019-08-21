load MNIST_Handwritten_data
maxepoch=20; %In the Science paper we use maxepoch=50, but it works just fine. 
numhid=500; 
[numcases numdims numbatches]=size(batchdata);
restart=1;

epsilonw      =0.1;   % Learning rate for weights 
epsilonvb     =0.1;   % Learning rate for biases of visible units 
epsilonhb     =0.1;   % Learning rate for biases of hidden units 
 weightcost  = 0.0002;    
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases,numdims, numbatches]=size(batchdata);

if restart ==1,
  restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.1*randn(numdims, numhid);vmin=min(min(vishid));
%   vmax=max(max(vishid));
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);
%   a=unique(vishid);
  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end
MAE1=[];

for epoch = epoch:maxepoch,
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches,
%  fprintf(1,'epoch %d batch %d\r',epoch,batch); 

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = batchdata(:,:,batch);
poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));    
batchposhidprobs=poshidprobs;

posprods    = data' * poshidprobs;
poshidact   = sum(poshidprobs);
posvisact = sum(data);

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% poshidstates = poshidprobs > rand(numcases,numhid);
poshidstates = poshidprobs > rand(size(poshidprobs,1),size(poshidprobs,2));
%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates *vishid' - repmat(visbiases,numcases,1)));
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    
  negprods  = negdata'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err= ((sum(sum( (data-negdata).^2)))/numcases) ;
errsum = err + errsum;
   if epoch>5
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;
vishidinc =epsilonw* (posprods-negprods)/numcases;


%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

 end
  errsum=sqrt(errsum);
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
  
  MAE1=[MAE1;errsum];
end
output=[];
 for ii=1:100
  output = [output data(ii,1:end)'  negdata(ii,:)'];
 end
   if epoch==1 
   close all 
%    figure('Position',[100,600,1000,200]);
   else 
   figure(1)
   end 
   mnistdisp(output);
   drawnow;
%%%%%%%%SVM%%%%%%% 
[numcases1,numdims1]=size(traindata);
[numcases2,numdims2]=size(testdata);
poshidprobs = 1./(1 + exp(-traindata*vishid - repmat(hidbiases,numcases1,1)));
tposhidprobs = 1./(1 + exp(-testdata*vishid - repmat(hidbiases,numcases2,1)));
train_set=poshidprobs;
test_set=tposhidprobs;
[mtrain,ntrain] = size(train_set);
[mtest,ntest] = size(test_set);
test_dataset = [train_set;test_set];
[dataset_scale,ps] = mapminmax(test_dataset',0,1);%y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin

train_set_labels;
test_set_labels;
model = svmtrain(train_set_labels, train_set,'-c 10,-g 0.01 ');
[predict_label, accuracy, prob_estimates] = svmpredict(test_set_labels, test_set, model);  
