function visualize_3d(frame_1, frame_2, P)

% Parameter to trade off between distribution and background
alpha = 0.5;

% Validate input
assert(all(size(frame_1) == size(frame_2)));
[nx, ny, nz] = size(frame_1);
n_pixels = nx * ny * nz;
assert(all(size(P) == [n_pixels, n_pixels]));

% Initialize plots
figure();
ax1 = subplot(311);
ax2 = subplot(312);
ax3 = subplot(313);

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
    title('frame 1');

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
