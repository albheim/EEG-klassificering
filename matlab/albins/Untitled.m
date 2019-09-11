close all;

% load acc
load('../test classification/acc16.mat');
rawsvm = squeeze(acc(:, 1, :));
rawlda = squeeze(acc(:, 2, :));
rawtree = squeeze(acc(:, 3, :));
rawbaggedtree = squeeze(acc(:, 4, :));
rawboostedtree = squeeze(acc(:, 5, :));

accnn = csvread('baseline_scores_first5.csv')';
rawcnn = accnn;
accnn = csvread('baseline_scores_25_first.csv')';
rawcnn(:, 6:30) = accnn;
accnn = csvread('baseline_scores_25_second.csv')';
rawcnn(:, 31:55) = accnn;
accnn = csvread('baseline_scores_45.csv')';
rawcnn(:, 56:100) = accnn;


% only 5 first values here, rest are running

cspmlp = csvread('csp_mlp_scores.csv');
cspsvm = csvread('csp_svm_scores.csv');
csplda = csvread('csp_lda_scores.csv');


d = mean(rawcnn);
mean(d)
std(d)
hist(d)
title('Random labels, raw data with 1D convolutional network')
% for i = 1:18
%     % change this to the data you want to see
%     d = squeeze(rawcnn(i, :));
%     hist(d)
%     title(sprintf("avg=%d", mean(d)))
%     pause
% end

addpath('normalitytest/')
Res = normalitytest(d);