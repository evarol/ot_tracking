%%% Color visualization of optimal transport tracking on synthetic data

clearvars -except red;
close all; clc;

%% Parameters

% Threshold for images
%IMG_THRESHOLD = 1.4;

% Time point to start at
T_TRACK = 7;

% Entropic regularization parameter
LAMBDA = 1.0;

% Maximum displacement allowed for transport
MAX_DISP = 20;


%% Load data

% Load synthetic data from file
load('/Users/cmcgrory/paninski_lab/worm/data/gmm_data_2d.mat');
V = data;

%% Preprocessing

% Threshold signal
%V(V < IMG_THRESHOLD) = 0;


%% Run OT

frame_1 = V(:, :, T_TRACK)';
frame_2 = V(:, :, T_TRACK + 1)';

% Run optimal transport on two successive frames
[P, ~] = optimal_transport(frame_1, frame_2, LAMBDA, MAX_DISP);


%% Visualize

visualize_2d(frame_1, frame_2, P, 0.3);