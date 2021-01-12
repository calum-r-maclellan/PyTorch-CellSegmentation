
% Summary: Script for taking in binary image of nuclei GTs, finding their centres,
% and projecting a circle of S*r onto the binary image. 
% We then want to add together these circles to get the rank for all edge pixels
% in the GT.

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

% Intuition: an edge pixel might be obscured by 5 cells that are overlapped.
% This will be very hard for the network to see, since what it expects to
% be a cytoplasm edge (the bit bordering the cell) is in fact a lot darker
% and mixed in with other bits of the cell. To help it figure out that the
% pixel is indeed the same thing as the outside (i.e. part of the same edge)
% we need to teach it to focus its efforts more on those hidden edges 
% (which are harder) instead of learning only the outer edges (which are easier). 
% Otherwise, the network will assume they are both the same on the basis of
% the easier one, and never actually learn what a 'hard' edge is. 
% By assigning weight/rank to each edge pixel, we can teach the network increase
% its learning capacity on harder edge pixels. 

% Started: 16/03/2020
% Finished: 17/03/2020

% Author: calmac

%% Procedure:
% 1. Get centroids and diameter of each nuclei in the GT image
% To do this, load the nuclei GTs and use regionprops() to get stats. on
% the centroids and diameter of circles with equivalent areas to nuclei.
% We also want to then find what the diameter of our search circle should
% be. Here I use the median of all the nuclei diameters.

% 2. Create search area circles and plot on 0 map.
% Use the median nuclei diameter and scale by 6 (arbitrary).

% 3. Create binary images of each search area, and sum to get rank values.
% In order for us to compute the rank values, we need to create
% intersecting regions between these search circles. This can be done by adding them
% together to produce a 'nuclei search area', where values from 1 to 5ish
% are produced when search circles intersect. 

% 4. Combine circles with GT cyto edges to assign ranks to edge pixels.
% We want to label the edge pixels only, and to do this I simply multiplied
% the GT edgemap with the ranked map. This way, pixels in the GT would be
% scaled according to the rank value at the same pixel position. We now
% have the ranked edge map for using in the new ranking loss function, 
% meaning that pixels have been weighed according to how hard they are to 
% detect, as we wanted.

% 5. EXTRA: create colourmap to highlight value at which edges are ranked.
% The last bit is for helping visualise the method. What it does is take
% the ranked edge map

clear; clc; close all;

%% Load GTs 
load('sample_targets.mat')

% Get an example
n_img = 90;
bw = targets{n_img}{3,1};
figure, imshow(bw);

%% 1. Get stats of each nuclei
% Centroid coordinates and diameter of circles with equivalent areas to
% nuclei
stats = regionprops('table', bw, 'Centroid', 'EquivDiameter');
mean_diameter = mean(stats.EquivDiameter);

% Use median value of diameter as the base diameter for creating a search
% circle
med_diameter = median(stats.EquivDiameter);

% Now estimate a search area of S * med_diameter, where S is an arbitrary
% scaling factor for enlarging the circles around each of the nuclei.
search_diameter(1:size(stats,1)) = round(6 * med_diameter);

%% 2. Next step: do it manually to get binary images 

% Get coordinates of centroids 
n = size(stats,1); % number of nuclei
nuc_centroids = zeros(n, 2);
for i = 1: n
    nuc_centroids(i,:) = stats.Centroid(i,:);
end

% Now take one at a time, plot the circle, get its binary image, 
% then store it into a cell array for adding later.
radius = search_diameter(1) / 2;
img = zeros(512, 512);
for ii = 1: n
    fig = imshow(img); % plot initial b/w image for overlaying binary images
    hold on;
    centroid = nuc_centroids(ii, :);
    x = centroid(1); y = centroid(2);
    [xunit, yunit] = get_circle(x, y, radius);
    circle = drawcircle('Center', centroid, 'Radius', radius);
    BW = createMask(circle);
    search_masks{ii, 1} = BW; % store BW image for that search area for adding later
    close all;
end

%% 3. Get ranked image
% Now that we have the binary image of each search area mask, we can now
% add them together to get the ranked map for that image. 
% Start with first two masks.
nuclei_search = imadd(search_masks{1,1}, search_masks{2,1} );
    
% Now repeat for remaining search areas
for i = 3: n
    n_mask = double(search_masks{i,1}); 
    nuclei_search = imadd(nuclei_search, n_mask);
end

% Show the search areas together
figure, imshow(nuclei_search);

%% Multiply with edge map to get ranked edge pixels
cyto_edges = double(targets{n_img}{2,1});
ranked_edges = immultiply(cyto_edges, nuclei_search);
figure, imshow(cyto_edges);
figure, imshow(ranked_edges); 

% %Got ranked edges, now need to complete edge map
% Do this by getting the ranked edges (intersecting edges),
% non-intersecting edges, and then add them together
% outer_edges = imsubtract(cyto_edges, ranked_edges);
% complete_ranked_map = imadd(outer_edges, ranked_edges);
% figure, imshow(outer_edges); 
% figure, imshow(complete_ranked_map);

%% Check to make sure it works
% max(nuclei_search(:))
% max(ranked_edges(:))
% max(complete_ranked_map(:))

%% 4. For some reason MATLAB is being gay.
% Doesnt let you keep the rank values when adding the outer and inner edges
% together (line 79), but instead converts them all to binary values. 
% Need a way of building up the ranked_edges (which has the correct rank
% values), so that it has the outer edges (rank=1) too.
for ii = 1: 262144 % for all the pixels in the ranked edges map   
    % if the pixel at ii is 0 in ranked_edges BUT is 1 in the GT
    if ranked_edges(ii) == 0 && cyto_edges(ii) == 1
        ranked_edges(ii) = 1; % then make that pixel = 1 in ranked_edges.
    else 
        continue;
    end
end
% this builds up the missing edges in ranked_edges, making them = 1 as we
% wanted.

figure, imshow(ranked_edges); 
max_rank = max(ranked_edges(:));
fprintf('Max value in ranked_edges: %d \n', max_rank)

% DONE!

%% Produce coloured binary image
% To make it easier to see where the ranked values are labelled.

rank0 = [0 0 0];      % black:  for rank=0
rank1 = [0 0 1];      % blue:   for rank=1
rank2 = [0 1 0];      % green:  for rank=2
rank3 = [1 0 1];      % purple: for rank=3
rank4 = [255/255 165/255 0];    % orange: for rank=4
rank5 = [1 0 0];      % red:    for rank=5

rankmap_colours = [rank0; rank1; rank2; rank3; rank4; rank5];
labels = {'Background','Rank=1','Rank=2','Rank=3','Rank=4','Rank=5'};
rankmap = rankmap_colours(1:1+max_rank, :);
labels = labels(1:1+max_rank);

figure, imagesc(ranked_edges); 
colormap(rankmap);
axis image
axis off
h = colorbar('Location','SouthOutside'); 
ylabel(h, 'Rank Map');
ro = (1 - 1/6)/2;
h.YTick = [ro 3*ro 5*ro 7*ro 9*ro 11*ro];
h.YTickLabel = {'Background','1','2','3','4','5'};

%% Circle function 
function [xunit, yunit] = get_circle(x, y, r)
    th = 0: pi/50: 2*pi;
    xunit = r .* cos(th) + x;
    yunit = r .* sin(th) + y;
end

