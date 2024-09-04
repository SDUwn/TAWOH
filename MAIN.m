%% init the workspace
close all; clear; clc; warning off;
addpath(genpath('./vlfeat/toolbox/'));
addpath(genpath('./Method/'));
addpath(genpath('./util/'));
run vl_setup;
%% globel settings
train_param.current_bits=32;
%% load dataset
train_param.ds_name='MIRFlickr-clip';   
train_param.N=16000;%MIRFlickr  
train_param.chunk_size=2000;
train_param.query_size=1000;
train_param.normalizeX = 1;
train_param.kernel = 0;
train_param.unsupervised=0;
[train_param,XTrain,LTrain,LLTrain,HLTrain,YTrain,XTrain_clip,YTrain_clip,LTrain_clip,XQuery,LQuery,LLQuery,XQuery_clip] = load_dataset(train_param);

%%
train_param.alpha=1e3;%1e2
train_param.beta = 1e3;%1e1
train_param.kesi = 1e1;%1e5
train_param.yita = 1e-3 ;%1e7
train_param.sigma=1e-1;%1e-3
train_param.phi = 1e4;%1e5
train_param.lamda = 1e5;%1e1
train_param.iter = 7;
[eva,t]=evaluate_it(XTrain_clip,YTrain_clip,LTrain,XQuery_clip,LQuery,train_param);

