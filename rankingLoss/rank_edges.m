function  ranked_edges = rank_edges(bw_nuclei, bw_cyto, S)

% Describe: function for producing the ranked edge map needed for the
% ranking focal loss. Designed to assign higher weight/rank to more
% difficult edge pixels for enhancing network learning, and thus prediction
% performance. 

% Input: bw image of nuclei GTs

% Output: bw edge map of cyto edges ranked according to overlap/difficulty
% level.

%------------------

%% Get stats of nuclei GTs
% Centroid coordinates and diameter of circles with equivalent areas to
% nuclei
stats = regionprops('table', bw_nuclei, 'Centroid', 'EquivDiameter');

% Use median value of diameter as the base diameter for creating a search
% circle
med_diameter = median(stats.EquivDiameter);

% Now estimate a search area of S * med_diameter, where S is an arbitrary
% scaling factor for enlarging the circles around each of the nuclei.
%S = 10; % 6 seems to work pretty well.
search_diameter(1:size(stats,1)) = round(S * med_diameter);

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
    imshow(img); % plot initial b/w image for overlaying binary images
    hold on;
    centroid = nuc_centroids(ii, :);
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

% Multiply with edge map to get ranked edge pixels
cyto_edges = double(bw_cyto);
ranked_edges = immultiply(cyto_edges, nuclei_search);

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

end