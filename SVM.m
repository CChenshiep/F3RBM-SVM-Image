
load MNIST_Handwritten_data
 train_set=traindata;
test_set=testdata;
%  test_set=Test_data;

[mtrain,ntrain] = size(train_set);
[mtest,ntest] = size(test_set);
test_dataset = [train_set;test_set];
[dataset_scale,ps] = mapminmax(test_dataset',0,1);%y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
% [train_set,ps]=mapminmax(train_set',0,1);train_set=train_set';

% [test_set,ps]=mapminmax(test_set',0,1);test_set=test_set';
%  dataset_scale = dataset_scale';
% train_set_labels =Ftrain_data(:,end);
% test_set_labels=Ftest_labels;

% model = svmtrain(train_set_labels, train_set,cmd);%c为惩罚因子；g为核函数半径的参数
%  model = svmtrain(train_set_labels, train_set,'-c 200,-kernel poly ');
 model = svmtrain(train_set_labels, train_set,'-c 10,-g 0.01 ');

% [predict_label] = svmpredict(test_set_labels, test_set', model);
[predict_label, accuracy, prob_estimates] = svmpredict(test_set_labels, test_set, model);

