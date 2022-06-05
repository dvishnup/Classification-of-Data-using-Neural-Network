clc
clear all
close all

sa1704 = readmatrix('data.xls','Sheet','sa1704');
sa0907 = readmatrix('data.xls','Sheet','sa0907');
ec1104 = readmatrix('data.xls','Sheet','ec1104');
ec1404 = readmatrix('data.xls','Sheet','ec1404');
sm1310 = readmatrix('data.xls','Sheet','sm1310');


figure,plot(sa1704), title('sa1704')
figure,plot(sa0907), title('sa0907')
figure,plot(ec1104), title('ec1104')
figure,plot(ec1404), title('ec1404')
figure,plot(sm1310), title('sm1310')


figure,plot(sa0907, 'c-x'), title('Combined data')
hold on
plot(sa1704, 'k-d'), title('Combined data')
plot(ec1104, 'y-s'), title('Combined data')
plot(ec1404, 'g-p'), title('Combined data')
plot(sm1310, 'm-<'), title('Combined data')
hold off

figure,mesh(sa0907), title('sa0907')
figure,mesh(sa1704), title('sa1704')
figure,mesh(ec1104), title('ec1104')
figure,mesh(ec1404), title('ec1404')
figure,mesh(sm1310), title('sm1310')

trainednet = feedforwardnet([20 10]);
view(trainednet);

data =sa1704;
data2=sa0907;
data3=ec1104;
data4=ec1404;
data5=sm1310;

%To improve efficiency we can enable or disable below function and adjust epochs 

%Change divideFcn: 'dividerand' to 'dividetrain' or ' '
%trainednet.divideFcn='dividetrain';
%change learning function from 'trainlm' to 'traingd'
%trainednet.trainFcn='trainlm';
%change the epochs when necessary
%trainednet.trainParam.epochs = 1000;

%remove outliers
for r=1:37 
    row = data(r,:);
    average = mean(row);
    deviation = std(row);
    threshold = [average-2*deviation average+2.1*deviation];
    for c=1:length(row) 
        if row(c) < threshold(1) 
            data(r,c)=average 
            if row(c) > threshold(2) 
                data(r,c) = average;
            end
        end
    end
end
figure,plot(data),title('Data set after outlier removal - SA1704')


for r=1:37 
    row = data2(r,:);
    average = mean(row);
    deviation = std(row);
    threshold = [average-2.1*deviation average+2.1*deviation];
    for c=1:length(row)
        if row(c) < threshold(1) 
            data2(r,c)=average 
            if row(c) > threshold(2) 
                data2(r,c) = average;
            end
        end
    end
end
figure,plot(data2),title('Data set after outlier removal - SA0907')
 
for r=1:37 
    row = data3(r,:);
    average = mean(row);
    deviation = std(row);
    threshold = [average-1.5*deviation average+2*deviation]; 
    for c=1:length(row) 
        if row(c) < threshold(1) 
            data3(r,c)=average
            if row(c) > threshold(2) 
                data3(r,c) = average;
            end
        end
    end
end
figure,plot(data3),title('Data set after outlier removal -EC1104 ')

for r=1:37 
    row = data4(r,:);
    average = mean(row);
    deviation = std(row);
    threshold = [average-2.2*deviation average+2*deviation]
    for c=1:length(row) 
        if row(c) < threshold(1) 
            data4(r,c)=average 
            if row(c) > threshold(2)
                data4(r,c) = average;
            end
        end
    end
end
figure,plot(data4),title('Data set after outlier removal -EC1404 ')

for r=1:37
    row = data5(r,:);
    average = mean(row);
    deviation = std(row);
    threshold = [average-2*deviation average+3*deviation]; 
    for c=1:length(row)
        if row(c) < threshold(1) 
            data5(r,c)=average
            if row(c) > threshold(2)
                data5(r,c) = average;
            end
        end
    end
end
figure,plot(data5),title('Data set after outlier removal -SM1310 ')

% Generate target data
sa0907_target = [];
for i = 1:47
 sa0907_target = [sa0907_target; 1 0 0 0 0];
end

sa1704_target = [];
for i = 1:50
 sa1704_target = [sa1704_target; 0 1 0 0 0 ];
end

ec1104_target = [];
for i = 1:50
 ec1104_target = [ec1104_target; 0 0 1 0 0 ];
end

ec1404_target = [];
for i = 1:40
 ec1404_target = [ec1404_target; 0 0 0 1 0];
end

sm1310_target = [];
for i = 1:50
 sm1310_target = [sm1310_target; 0 0 0 0 1];
end

% test  train and target
I=1:5:50;

sa0907_test = data2(:,I);
sa0907_train=data2;
sa0907_train(:,I)=[];
sa0907_target=sa0907_target';
sa0907_test_target=sa0907_target(:,[1:10]);

sa1704_test = data(:,I);
sa1704_train=data;
sa1704_train(:,I)=[];
sa1704_target=sa1704_target';
sa1704_test_target=sa1704_target(:,[1:10]);

ec1104_test =data3(:,I);
ec1104_train=data3;
ec1104_train(:,I)=[];
ec1104_target=ec1104_target';
ec1104_test_target=ec1104_target(:,[1:10]);

ec1404_test =data4(:,I);
ec1404_train=data4;
ec1404_train(:,I)=[];
ec1404_target=ec1404_target';
ec1404_test_target=ec1404_target(:,[1:10]);

sm1310_test =data5(:,I);
sm1310_train=data5;
sm1310_train(:,I)=[];
sm1310_target=sm1310_target';
sm1310_test_target=sm1310_target(:,[1:10]);

%Combine train data and test data
traindata = [sa1704_train,sa0907_train,ec1104_train,ec1404_train,sm1310_train];
traintarget = [sa1704_target,sa0907_target,ec1104_target,ec1404_target,sm1310_target]; 
testdata  = [sa1704_test,sa0907_test,ec1104_test,ec1404_test,sm1310_test];
testtarget=[sa1704_test_target,sa0907_test_target,ec1104_test_target,ec1404_test_target,sm1310_test_target];

%Train
init(trainednet);
trainednet = train(trainednet, traindata, traintarget);

%Testing using trained neural network with unseen data
res = sim(trainednet, testdata);
%view(trainednet)
plotconfusion( testtarget,res);
res_check = round(res);
