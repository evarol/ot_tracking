%%% Script for creating synthetic, 2D mixture-of-Gaussians data

%% Parameters 

% File to write to
OUT_FPATH = '/Users/cmcgrory/paninski_lab/worm/data/gmm_data_2d.mat';

% Image size 
IMG_SIZE = [100, 50];

% Image size limits
IMG_XLIM = [1, 100];
IMG_YLIM = [1, 50];

% Number of samples
T = 50;

% Sample rate (Hz)
SMP_RATE = 10;

% Number of mixture components
K = 15;

% Number of 'cycles' spanning worm
N_CYCLES = 2;

% Frequency of worm movement (Hz)
FREQ = 0.5;

% Amplitude of worm movement (image units)
AMP = 12.5;

% Scale of isotropic covariance matrix for GMM
COV_SCL = 5.0;

% Flag for whether or not to add noise
ADD_NOISE = false;

% Noise level (stddev of Gaussian noise)
NOISE_STD = 1e-4;


%% Create time series of mean positions

% X-values of means are equally spaced; don't change in time
means_x = linspace(IMG_XLIM(1), IMG_XLIM(2), K + 2);
means_x = means_x(2:K+1);
means_x = repmat(means_x, [T, 1]);

% Y-values of means oscillate in time
phases = linspace(0, N_CYCLES * 2 * pi, K);
offset = IMG_YLIM(1) + (IMG_YLIM(2) - IMG_YLIM(1)) / 2;
rads = (2 * pi * FREQ / SMP_RATE) * (0:(T-1));
means_y = offset + AMP * sin(rads' + phases);

%% Plot motion of means in real-time
% 
% for t = 1:T
%     
%     scatter(means_x(t, :), means_y(t, :));
%     xlim(IMG_XLIM);
%     ylim(IMG_YLIM);
%     
%     pause(1 / SMP_RATE);
%     
% end

%% Create series of GMM densities using mean poositions

% Covariance matrix is isotropic, with scale determined by parameter
sigma = COV_SCL * eye(2);

% Create grid for evaluating densities on
xs = linspace(IMG_XLIM(1), IMG_XLIM(2), IMG_SIZE(1));
ys = linspace(IMG_YLIM(1), IMG_YLIM(2), IMG_SIZE(2));
[yg, xg] = meshgrid(ys, xs);
grid = [xg(:), yg(:)];

data = zeros(IMG_SIZE(1), IMG_SIZE(2), T);
for t = 1:T
    
    % Collect means for time t into vector
    mu = [means_x(t, :)', means_y(t, :)'];
    
    % Create GM distribution with equal mixing proportions
    gm = gmdistribution(mu, sigma);
    
    pdf_vals = pdf(gm, grid);
    data(:, :, t) = reshape(pdf_vals, IMG_SIZE);
end

%% If flag is set, add noise to data

if ADD_NOISE 
    for t = 1:T
        
        % Add Gaussian noise to image
        img_nn = data(:, :, t) + NOISE_STD * randn(IMG_SIZE);
        
        % Set all negative pixel values to zero (data must be non-negative)
        img_nn(img_nn < 0) = 0;
        
        % Normalize image to make it valid distribution
        data(:, :, t) = img_nn ./ sum(img_nn(:));
    end
end
   
%% Plot dataset

for t = 1:T
    
    imagesc(data(:, :, t)');
    xlim(IMG_XLIM);
    ylim(IMG_YLIM);
    
    pause(1 / SMP_RATE);
    
end

%% Write data to file

save(OUT_FPATH, 'means_x', 'means_y', 'data');