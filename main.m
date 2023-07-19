clc
clear 
close all

%%Defining Input and Output Space %%
% X = number of subject x 1 cell array
% X{.,1} = number of instances x number of features array
% Y = 1 x number of raters cell array
% Y{1,.} = 1 x number of subject cell array
% Y{1,.}{1,.} = number of instances x 1 array


% X = INPUT SPACE
% Y = OUTPUT SPACE
% task_num = number of raters

%% Example
% X{1,1}=randn(3,32);
% 
% X{2,1}=randn(4,32);
% 
% X{3,1}=randn(5,32);
% 
% X{4,1}=randn(6,32);
% 
% 
% %task_num = 4
% 
% Y{1,1}{1,1}=[-1 -1 1]';
% Y{1,1}{1,2}=[-1 NaN 1 -1]';
% Y{1,1}{1,3}=[-1 1 1 -1 -1]';
% Y{1,1}{1,4}=[1 1 1 NaN -1 -1]';
% 
% Y{1,2}{1,1}=[-1 1 1]';
% Y{1,2}{1,2}=[-1 NaN 1 -1]';
% Y{1,2}{1,3}=[-1 1 1 -1 -1]';
% Y{1,2}{1,4}=[1 1 1 -1 -1 -1]';
% 
% Y{1,3}{1,1}=[NaN NaN 1]';
% Y{1,3}{1,2}=[-1 1 1 -1]';
% Y{1,3}{1,3}=[-1 NaN 1 -1 -1]';
% Y{1,3}{1,4}=[1 1 1 -1 -1 -1]';
% 
% 
% Y{1,4}{1,1}=[NaN 1 1]';
% Y{1,4}{1,2}=[-1 1 1 -1]';
% Y{1,4}{1,3}=[NaN 1 1 -1 -1]';
% Y{1,4}{1,4}=[1 1 1 -1 -1 -1]';
% 

%% Code

R = eye (task_num) - ones (task_num) / task_num;  %regularized MTL penalty

results=SRMTL(X, Y, R, task_num);