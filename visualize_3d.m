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


%% Plot first frame

% Initialize plots
ax1 = subplot(311);
ax2 = subplot(312);
ax3 = subplot(313);

% Parameter to trade off between distribution and background
alpha = 0.5;

% Compute max projections for both frames
[mp_1, zmax_1] = max(frame_1, [], 3);
mp_2 = max(frame_2, [], 3);

% Compute min and max for image
img_min = min([frame_1(:); frame_2(:)]);
img_max = max([frame_1(:); frame_2(:)]);

% Create RGB image from image max. projection
img1_rgb = alpha * (repmat(mp_1, [1, 1, 3]) - img_min) ./ (img_max - img_min);

% Plot first frame
subplot(ax1);
imshow(img1_rgb, 'InitialMagnification', 'fit');
title(sprintf('frame: %d', T_TRACK));


%% Get point from user and plot pushforward

% TODO: Make this endless loop
while (1 == 1)

    % Get XY coordinates of point (order getpts() returns is reversed)
    [pts_y, pts_x] = getpts(ax1);
    assert(size(pts_y, 1) == 1);
    assert(size(pts_x, 1) == 1);
    
    % Round to nearest pixel
    pt_y = ceil(pts_y(1));
    pt_x = ceil(pts_x(1));

    % Z-coordinate is coordinate of max value at XY point
    pt_z = zmax_1(pt_x, pt_y);
    
    % Mark selected pixel in green
    img1_rgb_mk = img1_rgb;
    img1_rgb_mk(pt_x, pt_y, 1) = 0;
    img1_rgb_mk(pt_x, pt_y, 2) = 1;
    img1_rgb_mk(pt_x, pt_y, 1) = 0;
    
    % (re)-plot first frame
    subplot(ax1);
    imshow(img1_rgb_mk, 'InitialMagnification', 'fit');
    title(sprintf('frame: %d', T_TRACK));

    % Get pushforward distribution of selected pixel
    pt_idx = sub2ind([nx, ny, nz], pt_x, pt_y, pt_z);
    dist_nn = P(:, pt_idx);
    dist_vec = dist_nn ./ sum(dist_nn);
    
    % Create 2D max. projection of distribution
    dist_img = reshape(dist_vec, [nx, ny, nz]);
    dist_mp = max(dist_img, [], 3);
    
    % Scale distribution so that mode is equal to one, and apply square root
    dist_scl = sqrt(dist_mp ./ max(dist_mp(:)));
    
    % Create RGB image with distribution superimposed on frame
    img2_rgb = alpha * (repmat(mp_2, [1, 1, 3]) - img_min) ./ (img_max - img_min);
    img2_rgb(:, :, 1) = img2_rgb(:, :, 1) + (1 - alpha) * dist_scl;
    
    % Mark selected pixel in green
    img2_rgb(pt_x, pt_y, 1) = 0;
    img2_rgb(pt_x, pt_y, 2) = 1;
    img2_rgb(pt_x, pt_y, 1) = 0;
    
    % Clear previous distribution plots
    cla(ax2, 'reset');
    cla(ax3, 'reset');
    
    % Plot second frame with distribution
    subplot(ax2);
    imshow(img2_rgb, 'InitialMagnification', 'fit');
    title(sprintf('frame: %d', T_TRACK + 1));
    
    % Create RGB image with distribution alone
    img3_rgb = zeros(size(dist_scl, 1), size(dist_scl, 2), 3);
    img3_rgb(:, :, 1) = (1 - alpha) * dist_scl;
    
    % Mark selected pixel in green
    img3_rgb(pt_x, pt_y, 1) = 0;
    img3_rgb(pt_x, pt_y, 2) = 1;
    img3_rgb(pt_x, pt_y, 1) = 0;

    % Plot distribution alone
    subplot(ax3);
    imshow(img3_rgb, 'InitialMagnification', 'fit');
    title('pushforward distribution');
    
end
