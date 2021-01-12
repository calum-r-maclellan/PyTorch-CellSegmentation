% main.m
% Use rank_edges.m for automating ranked edges across entire training set.

% Describe: Script for taking in binary image of nuclei GTs, finding their centres,
% and projecting a circle of S*r onto the binary image. 
% We then want to add together these circles to get the rank for all edge pixels
% in the GT. REPEAT FOR ALL IMAGES IN TRAINING dataset.

% Purpose: to create a wee program that figures out the bits of the
% cell clump that are the hardest to predict. The reason for doing this is
% so that we can tell the network the bits it has to learn the most, for
% improving its ability to detect even the most obscure cytoplasm edges. 

% Method: use the nuclei to find the areas in the cell clump where edges 
% are obscured by overlapping. The amount of weight (which we call 'rank')
% that we assign is proportional to the amount of search circles that are
% overlapping that particular edge. When a circle overlaps another, they
% are added together (i.e. 1 + 1 = rank2). The more circles that overlap,
% the higher the rank, therefore the more difficult that edge is to detect, 
% therefore it has higher impact on the loss function, therefore the network 
% learns it more. 

% Started: 17.03.2020. 
% Complete: .

% Author: calmac
%-------------------

clear; clc; close all;

%% Load training data
% Need both cytoplasm edge and nuclei GTs for this to work.
load('Volumes/CALUM_ENGD/ISBI_CellSegmentation/ranking-stuff/training_targets.mat') % 3x1 cells (x855): background, cyto, nuclei

%% Compute rank map for each image 
S = 7; % scaling factor. For setting diameter of searching circles.

for ii = 1: 45
    bw_nuclei = targets{ii}{3,1};
    bw_cyto = targets{ii}{2,1};
    ranked_maps{ii,1} = rank_edges(bw_nuclei, bw_cyto, S);
end
% takes about 5mins to complete 855 rank maps.

%% Check to see if they have been ranked 
% Colour ranked edges
ranked_edges = ranked_maps{44};

rank0 = [0 0 0];                % black:  for rank=0
rank1 = [0 0 1];                % blue:   for rank=1
rank2 = [0 1 0];                % green:  for rank=2
rank3 = [1 0 1];                % purple: for rank=3
rank4 = [0 165/255 0.6];        % cyan:   for rank=4
rank5 = [255/255 165/255 1];    % orange: for rank=5
rank6 = [255/255 200/255 0];    % orange: for rank=6
rank7 = [255/255 215/255 0.8];    % orange: for rank=7
rank8 = [1 0 0];                % red:    for rank=8(highest seen)

rankmap_colours = [rank0; rank1; rank2; rank3; rank4; rank5; rank6; rank7; rank8];
max_rank = max(ranked_edges(:));
rankmap = rankmap_colours(1:1+max_rank, :);
figure, imagesc(ranked_edges); 
colormap(rankmap);
axis image
axis off
h = colorbar('Ticks',[],...
         'TickLabels',{},'Location','SouthOutside'); 
ylabel(h, 'Rank Map');

%% Save if it works
save('training_rankmaps.mat','ranked_maps')


