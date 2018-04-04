clear; clc; close all

[X,Y,n] = aux_load('Visual','05');
N = sum(n); r = 8;

X = aux_extr(X,769:1024);
X = aux_chan(X,[5 24]);
%X = aux_svd(X,1);
X = aux_deci(X,r);
%X = aux_feat(X);

X = aux_prep(X);
acc = aux_eval(X,Y);