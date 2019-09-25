function visualize_2d(frame_1, frame_2, P, alpha)
% Visualize optimal transport (OT) between two 2D frames.
%
% Args:
%     frame_1 (2D array): 'Source' frame for OT
%     frame_2 (2D array): 'Target' frame for OT
%     P (2D array): Transport matrix computed by OT algorithm
%     alpha: Parameter to trade off between distribution and background

% Validate input
assert(all(size(frame_1) == size(frame_2)));
[nx, ny] = size(frame_1);
n_pixels = nx * ny;
assert(all(size(P) == [n_pixels, n_pixels]));

% Initialize plots
figure();
ax1 = subplot(311);
ax2 = subplot(312);
ax3 = subplot(313);

% Compute min and max for image
img_min = min([frame_1(:); frame_2(:)]);
img_max = max([frame_1(:); frame_2(:)]);

% Create RGB image from image max. projection
img1_rgb = alpha * (repmat(frame_1, [1, 1, 3]) - img_min) ...
    ./ (img_max - img_min);

% Plot first frame
subplot(ax1);
imshow(img1_rgb, 'InitialMagnification', 'fit');
title('frame 1');


% Get point from user and plot pushforward
while (1 == 1)

    % Get XY coordinates of point (order getpts() returns is reversed)
    [pts_y, pts_x] = getpts(ax1);
    assert(size(pts_y, 1) == 1);
    assert(size(pts_x, 1) == 1);
    
    % Round to nearest pixel
    pt_y = ceil(pts_y(1));
    pt_x = ceil(pts_x(1));

    % Mark selected pixel in green
    img1_rgb_mk = img1_rgb;
    img1_rgb_mk(pt_x, pt_y, 1) = 0;
    img1_rgb_mk(pt_x, pt_y, 2) = 1;
    img1_rgb_mk(pt_x, pt_y, 1) = 0;
    
    % (re)-plot first frame
    subplot(ax1);
    imshow(img1_rgb_mk, 'InitialMagnification', 'fit');
    title('frame 1');

    % Get pushforward distribution of selected pixel
    pt_idx = sub2ind([nx, ny], pt_x, pt_y);
    dist_nn = P(pt_idx, :);
    dist_vec = dist_nn ./ sum(dist_nn);
    dist_img = reshape(dist_vec, [nx, ny]);
    
    % Scale distribution so that mode is equal to one, and apply square root
    dist_scl = sqrt(dist_img ./ max(dist_img(:)));

    % Create RGB image with distribution superimposed on frame
    img2_rgb = alpha * (repmat(frame_2, [1, 1, 3]) - img_min) ...
        ./ (img_max - img_min);
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
    title('frame 2');
    
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
