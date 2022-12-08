% Make computationally generated patients using leukemia mechanotype data
% Using all qc-DC features
% Homogeneous healthy patients and hetereogenous patients

clc
clear


% Will be sampling random numbers with NO replacement, no repeated numbers
% in a patient
% Using randperm instead of randi (because randi allows for the possibility of repeated numbers)


%% Initialize 

% Choose how many cells will be in a sample from each synthetic patient
numCells = 100;
numPatients = 500;

% Variables for machine learning 
numModelsMonteCarl = 4;
fractionHoldOut = 0.2; %for training dataset and test datasets 

% Proportions of VDL-treated cells in the "Resistant" class
% Will hold machine learning accuracy for each unique model from a Monte
% Carlo iteration
groupsAccuracy = zeros(numModelsMonteCarl+1,7);
groupsAccuracy(1,:) = [1 2 10 25 50 75 99];



%prediLabels = zeros(numPatients*2*fractionHoldOut, 1, 7)



%% Load qc-DC mechanotyping information

% Update later - this can be pulled out from a different Excel sheet
LAX53 = readtable('LAX53.xlsx');
LAX53VDL = readtable('LAX53VDL.xlsx');
LAX7R = readtable('LAX7R.xlsx');
LAX7RVDL = readtable('LAX7RVDL.xlsx');


% Edit tables for only relevant features
LAX53 = removevars(LAX53, {'C1EntryFrame','C1ExitFrame','EntryOcclusion','ExitOcclusion','AverageOcclusion','MaxOcclusion','Lane','VideoName', ...
    'Experiment','Condition','C2ExitFrame', 'C3ExitFrame', 'C4ExitFrame', 'C5ExitFrame', 'C6ExitFrame', 'C1TotalTime', 'Rsquared', 'AdjustedRsquared'});
LAX53VDL = removevars(LAX53VDL, {'C1EntryFrame','C1ExitFrame','EntryOcclusion','ExitOcclusion','AverageOcclusion','MaxOcclusion','Lane','VideoName', ...
    'Experiment','Condition','C2ExitFrame', 'C3ExitFrame', 'C4ExitFrame', 'C5ExitFrame', 'C6ExitFrame', 'C1TotalTime', 'Rsquared', 'AdjustedRsquared'});
LAX7R = removevars(LAX7R, {'C1EntryFrame','C1ExitFrame','EntryOcclusion','ExitOcclusion','AverageOcclusion','MaxOcclusion','Lane','VideoName', ...
    'Experiment','Condition','C2ExitFrame', 'C3ExitFrame', 'C4ExitFrame', 'C5ExitFrame', 'C6ExitFrame', 'C1TotalTime', 'Rsquared', 'AdjustedRsquared'});
LAX7RVDL = removevars(LAX7RVDL, {'C1EntryFrame','C1ExitFrame','EntryOcclusion','ExitOcclusion','AverageOcclusion','MaxOcclusion','Lane','VideoName', ...
    'Experiment','Condition','C2ExitFrame', 'C3ExitFrame', 'C4ExitFrame', 'C5ExitFrame', 'C6ExitFrame', 'C1TotalTime', 'Rsquared', 'AdjustedRsquared'});



% Make pool of data
poolSens = vertcat(LAX53,LAX7R);
poolRes = vertcat(LAX53VDL,LAX7RVDL);

% Need to convert from table to cell array because some of the indexing
% works better in a cell array e.g., patientsHomog(numCells+1,:)
poolSens = table2cell(poolSens);
poolRes = table2cell(poolRes);

for cMdls = 1:numModelsMonteCarl
%% Generate computational sets, or "patients", with a random sample of cells

% randomizePure and randomizeMixed functions

%function [patientsPure] = randomizePure(poolSens, numCells, numPatients)
% Note: it's possible to quickly compare how many cells are re-used across
% different patients. Do this by changing the function, randomizePure, to
% include "iHomog" as an additional output. Then, compare for duplicate
% indices 
%
%function [patientPop] = randomizeMixed(poolSens, poolRes, numCells, numPatients, percentRes)


% Combine the pure and mixed populations. Partition for machine learning

% Note, can't combine at the beginning. Need to be able to know which
% "resistant" patients are from which proportion of mixed populations

% Combine pure populations and mixed populations before partitioning
% Convert cell back to table since it's easier to work with tables in the
% machine learning functions 
% Nomenclature of patientsComb# = patients Combined %Resistant

patientsPure1 = randomizeAllFeat_Pure(poolSens, numCells, numPatients);
patientsPure2 = randomizeAllFeat_Pure(poolSens, numCells, numPatients);
patientsPure3 = randomizeAllFeat_Pure(poolSens, numCells, numPatients);
patientsPure4 = randomizeAllFeat_Pure(poolSens, numCells, numPatients);
patientsPure5 = randomizeAllFeat_Pure(poolSens, numCells, numPatients);
patientsPure6 = randomizeAllFeat_Pure(poolSens, numCells, numPatients);
patientsPure7 = randomizeAllFeat_Pure(poolSens, numCells, numPatients);

%function [patientPop] = randomizeMixed(poolSens, poolRes, numCells, numPatients, percentRes)
patientsMixed99_1 = randomizeAllFeat_Mixed(poolSens, poolRes, numCells, numPatients, 0.01);
patientsMixed98_2 = randomizeAllFeat_Mixed(poolSens, poolRes, numCells, numPatients, 0.02);
patientsMixed90_10 = randomizeAllFeat_Mixed(poolSens, poolRes, numCells, numPatients, 0.10);
patientsMixed75_25 = randomizeAllFeat_Mixed(poolSens, poolRes, numCells, numPatients, 0.25);
patientsMixed50_50 = randomizeAllFeat_Mixed(poolSens, poolRes, numCells, numPatients, 0.5);
patientsMixed25_75 = randomizeAllFeat_Mixed(poolSens, poolRes, numCells, numPatients, 0.75);
patientsMixed1_99 = randomizeAllFeat_Mixed(poolSens, poolRes, numCells, numPatients, 0.99);


%% Combine the pure and mixed populations. Partition for machine learning

% Note, can't combine at the beginning. Need to be able to know which
% "resistant" patients are from which proportion of mixed populations

% Combine pure populations and mixed populations before partitioning
% Convert cell back to table since Classification Learner only accepts tables
% and doubles
% Nomenclature of patientsComb# = patients Combined %Resistant
patientsComb1 = cell2table(vertcat(patientsPure1', patientsMixed99_1'));
patientsComb2 = cell2table(vertcat(patientsPure2', patientsMixed98_2'));
patientsComb10 = cell2table(vertcat(patientsPure3', patientsMixed90_10'));
patientsComb25 = cell2table(vertcat(patientsPure4', patientsMixed75_25'));
patientsComb50 = cell2table(vertcat(patientsPure5', patientsMixed50_50'));
patientsComb75 = cell2table(vertcat(patientsPure6', patientsMixed25_75'));
patientsComb99 = cell2table(vertcat(patientsPure7', patientsMixed1_99'));


%% Complete 80%-20% holdout 
patientsPartition = cvpartition(height(patientsComb1),'Holdout',0.2); 

% Training data indexing 
iTrain = training(patientsPartition);

% Combine all varying proportions of mixed populations and pure populations here
patientsTrain = vertcat(patientsComb1(iTrain,:),patientsComb2(iTrain,:), patientsComb10(iTrain,:),patientsComb25(iTrain,:), patientsComb50(iTrain,:), patientsComb75(iTrain,:), patientsComb99(iTrain,:));

% Grab test data
iTest = test(patientsPartition);
patientsTest1 = patientsComb1(iTest,:);
patientsTest2 = patientsComb2(iTest,:);
patientsTest10 = patientsComb10(iTest,:);
patientsTest25 = patientsComb25(iTest,:);
patientsTest50 = patientsComb50(iTest,:);
patientsTest75 = patientsComb75(iTest,:);
patientsTest99 = patientsComb99(iTest,:);


%% Train model
% Hyperparameter optimization is completed using Bayesian optimization 

clear mdl 

tic

% kNN
% mdl = fitcknn(patientsTrain(:,1:end-1),patientsTrain(:,end), 'Standardize', true,'OptimizeHyperparameters',{'Distance', 'NumNeighbors', 'DistanceWeight'}, ...
%    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus', 'ShowPlots', 0, 'MaxObjectiveEvaluations', 20));

% SVM
% mdl = fitcsvm(patientsTrain(:,1:end-1),patientsTrain(:,end), 'Standardize', true, 'OptimizeHyperparameters','auto', ...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus', 'ShowPlots', 0, 'MaxObjectiveEvaluations', 10));


% Ensemble
mdl = fitcensemble(patientsTrain(:,1:end-1),patientsTrain(:,end), 'OptimizeHyperparameters',{'Method','NumLearningCycles','LearnRate'}, ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus', 'ShowPlots', 0, 'MaxObjectiveEvaluations', 10));
toc




%% Test model
prediLabels1 = predict(mdl, patientsTest1(:,1:end-1));
prediLabels2 = predict(mdl, patientsTest2(:,1:end-1));
prediLabels10 = predict(mdl, patientsTest10(:,1:end-1));
prediLabels25 = predict(mdl, patientsTest25(:,1:end-1));
prediLabels50 = predict(mdl, patientsTest50(:,1:end-1));
prediLabels75 = predict(mdl, patientsTest75(:,1:end-1));
prediLabels99 = predict(mdl, patientsTest99(:,1:end-1));

% prediCompare = table(predi, patientsTest1(:,end))

%Initialize class performance with labels
cp1 = classperf(table2cell(patientsTest1(:,end)));
cp2 = classperf(table2cell(patientsTest2(:,end)));
cp10 = classperf(table2cell(patientsTest10(:,end)));
cp25 = classperf(table2cell(patientsTest25(:,end)));
cp50 = classperf(table2cell(patientsTest50(:,end)));
cp75 = classperf(table2cell(patientsTest75(:,end)));
cp99 = classperf(table2cell(patientsTest99(:,end)));


%Calculate class performance including accuracy
classperf(cp1,prediLabels1);
classperf(cp2,prediLabels2);
classperf(cp10,prediLabels10);
classperf(cp25,prediLabels25);
classperf(cp50,prediLabels50);
classperf(cp75,prediLabels75);
classperf(cp99,prediLabels99);





groupsAccuracy(cMdls+1,1) = cp1.CorrectRate;
groupsAccuracy(cMdls+1,2) = cp2.CorrectRate;
groupsAccuracy(cMdls+1,3) = cp10.CorrectRate;
groupsAccuracy(cMdls+1,4) = cp25.CorrectRate;
groupsAccuracy(cMdls+1,5) = cp50.CorrectRate;
groupsAccuracy(cMdls+1,6) = cp75.CorrectRate;
groupsAccuracy(cMdls+1,7) = cp99.CorrectRate;

end

save groupsAccuracy 

% beep
% pause(2)
% beep
% pause(2)
% beep


% figure
% confusionchart(table2cell(patientsTest1(:,end)), predi)

%%
