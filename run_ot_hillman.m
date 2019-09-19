%%% Color visualization of optimal transport tracking

clearvars -except red;
close all; clc;

%% Parameters

% Subsampling factor
SCL_SUBSMP = 0.10;

% Threshold for images
IMG_THRESHOLD = 1.4;

% Time point to start at
T_TRACK = 50;

% Entropic regularization parameter
LAMBDA = 0.50;

% Maximum displacement allowed for transport
MAX_DISP = 10;


%% Load data

% Load RFP ('red') signal from Hillman data
if ~exist('red', 'var')
    load('/Users/cmcgrory/paninski_lab/worm/data/hillman_video.mat');
    red = permute(red, [2 1 3 4]);
end

%% Preprocessing

% Spatially subsample 'red' signal
nx = ceil(SCL_SUBSMP * size(red, 1));
ny = ceil(SCL_SUBSMP * size(red, 2));
nz = ceil(SCL_SUBSMP * size(red, 3));
nt = size(red, 4);
V = zeros(nx, ny, nz, nt);
for t = 1:nt
    V(:, :, :, t) = imresize3(red(:, :, :, t), SCL_SUBSMP);
end

% Threshold signal
V(V < IMG_THRESHOLD) = 0;


%% Run OT

frame_1 = V(:, :, :, T_TRACK);
frame_2 = V(:, :, :, T_TRACK + 1);

% Run optimal transport on two successive frames
[P, ~] = optimal_transport(frame_1, frame_2, LAMBDA, MAX_DISP);


%% Visualize

visualize_3d(frame_1, frame_2, P);