%%% Optimal transport tracking on 2D synthetic data

clearvars; close all; clc;

%% Parameters

% Threshold for images
IMG_THRESHOLD = 1e-4;

% Entropic regularization parameter
LAMBDA = 20.0;

% Maximum displacement allowed for transport
MAX_DISP = 20;

% Tradeoff between track distribution and background image
ALPHA = 0.5;


%% Load data

% Load synthetic data from file
load('/Users/cmcgrory/paninski_lab/worm/data/gmm_data_2d.mat');
V = data;
nx = size(V, 1);
ny = size(V, 2);
nt = size(V, 3);

% Compute min and max pixel values across entire video
img_min = min(V(:));
img_max = max(V(:));


%% Run OT for all frames, while tracking point

% Choose initial point
idx = 3;
x_init = ceil(means_x(1, idx));
y_init = ceil(means_y(1, idx));
px_init = sub2ind([nx, ny], x_init, y_init);

% Array for holding distributions at each tracking step
track_dist = zeros(nt - 1, nx * ny);

% Set first track distribution to point mass on init pixel
track_dist(1, px_init) = 1;

for t = 1:(nt - 1)
    
    % TODO: Make this sparse
    % Run OT on current frame and next frame
    [P, ~] = optimal_transport_sparse( ...
        V(:, :, t), V(:, :, t + 1), IMG_THRESHOLD, LAMBDA, MAX_DISP);  
    
    % TODO: Find cleaner way to do this!
    % Conditional form of transport matrix (T_ij = P(j | i))
    z = sum(P, 2);
    z(z == 0) = 1;
    T = P ./ z;
    
    % Compute pushforward of current tracking distribution
    track_dist(t + 1, :) = track_dist(t, :) * T;
   
end


%% Plot results

track_vid = zeros(nx, ny, nt, 3);

for t = 1:nt
    
    % Find mode of distribution
    dist = track_dist(t, :);
    [~, mode_idx] = max(dist);
    [mode_x, mode_y] = ind2sub([nx, ny], mode_idx);
    
    % Scale distribution so that mode is equal to one, and apply square root
    dist_img = reshape(dist, [nx, ny]);
    dist_scl = sqrt(dist_img ./ max(dist_img(:)));

    % Create RGB image with distribution superimposed on frame
    frame = (V(:, :, t) - img_min) ./ (img_max - img_min);
    img_rgb = ALPHA * repmat(frame, [1, 1, 3]);
    img_rgb(:, :, 1) = img_rgb(:, :, 1) + (1 - ALPHA) * dist_scl;
    
    % Mark mode pixel in green
    img_rgb(mode_x, mode_y, 1) = 0;
    img_rgb(mode_x, mode_y, 2) = 1;
    img_rgb(mode_x, mode_y, 3) = 0;
    
    track_vid(:, :, t, :) = img_rgb;
    
end

% Switch x and y dimensions for plotting
track_vid = permute(track_vid, [2, 1, 3, 4]);

imshow3d(track_vid);
