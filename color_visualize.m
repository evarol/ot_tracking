%%% Color visualization of optimal transport tracking

clearvars -except red;
close all; clc;

%% Parameters

% Time point to start at
tk = 50;

% Entropic regularization parameter
lambda = 0.5;


%% Load data

if ~exist('red', 'var')
    load('/Users/cmcgrory/paninski_lab/worm/data/hillman_video.mat');
    red = permute(red, [2 1 3 4]);
end

% V is the 'red' image, summed along the z-axis
nt = size(red, 4);
V = zeros(size(red, 1), size(red, 2), nt);
for t = 1:nt
    V(:, :, t) = sum(red(:, :, :, t), 3);
end

% Subsample V slices
for t = 1:nt
    tmp(:, :, t) = imresize(V(:, :, t), 0.24);
end
V = tmp; 
clear tmp;


%% Run OT

% Run optimal transport on two successive frames
[P, ~] = optimal_transport(V(:, :, tk), V(:, :, tk + 1), lambda, 100);


%% Track points

% Choose points to track
% TODO: Randomize this
nx = size(V, 1);
ny = size(V, 2);
pts = [ ...
    14, 75;
    5, 100;
    22, 23;
];

% Compute conditional distribution matrix from joint
P_cond = P ./ sum(P, 1);

% Get distribution of tracked points
pt_idxs = sub2ind([nx, ny], pts(:, 1), pts(:, 2));
dist_vals = P_cond(:, pt_idxs);


%% Plot tracked points and corresponding distributions

idx = 1;
pt = pts(idx, :);
dist = dist_vals(:, idx);

figure();

subplot(211);
pt_img = zeros(nx, ny);
pt_img(pt(1), pt(2)) = 1;
imagesc(pt_img);
colorbar;
title(sprintf('Random pixel (t=%d)', tk));
 
subplot(212);
dist_img = vec_to_img(dist, [nx, ny]);
imagesc(dist_img);
colorbar;
title(sprintf('Pushforward (t=%d)', tk + 1));

suptitle(['Optimal transport (\lambda=', sprintf('%.02f', lambda), ')']);