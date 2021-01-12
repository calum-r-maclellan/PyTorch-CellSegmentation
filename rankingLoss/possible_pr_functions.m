% Simple script for analysing the effects on the weight given by my new
% focal loss function when different functions are used for computing Pr,
% the ranked confidence. 
% This way, the computers confidence in a pixel belonging to an edge is
% related to the level of difficulty of obtaining that edge.
% For example, if r is high, then a highly confident prediction should 

% What we're wanting in this function is to have a much larger losses for
% when its the most difficult (r=5) and has the least confidence, whereas
% we want very little loss for when its the easiest (r=1) and has most
% confidence. 

% Started: 11/03/2020.
clear; clc; close all; 


%% Latest idea: use a low-pass filter.
% Idea: for ranking loss:
% - to get more learning: make pr smaller 
% - to get less learning: make pr larger
% since its (1 - pr)^2, so when pr smaller, that function gets larger and
% vice versa.

% To make it smaller/larger we need a gain function: use the one for a low
% pass filter so that small ranks (i.e r -> 0) give large gains.
% As above, if we get larger gains, pr gets bigger, so

% As the rank increases, what effect does this have on the gain?
figure;
% title('Effect of rank on gain for various powers, a');
xlabel('r');
ylabel('G(r)');
hold on            
for a = 1: 6
    r = 0: 0.1: 8;
    Gain = 1 ./ sqrt(1 + r.^a);
    plot(r, Gain, '-', 'LineWidth', 1.5);
    Legend_a{a}=strcat('a=', num2str(a));
    hold on 
end
legend(Legend_a, 'Location','Best');
hold off;

%% Effect of rank on assigned weight for different values of p

figure;
xlabel('r');
ylabel('L_{rank} (i.e. weight assigned to pixel)');
hold on
i=1;
for p = 0.1: 0.1: 1
    r = 0: 0.1: 8;
    Gain = 1 ./ (sqrt(1 + r.^2));
    pr = p .* Gain;    
    RL = - (1 - pr).^2 .* log(pr);
    plot(r, RL, '-', 'LineWidth', 1.5);
    Legend_p{i}=strcat('p=', num2str(p));
    i = i+1;
    hold on;
end
legend(Legend_p, 'Location','Best');
hold off;

%% As the confidence increases, what effect does this have on the gain for different values of r?
figure;
xlabel('Network output (p)');
ylabel('L_{rank}');
hold on            
for r = 1: 6
    p = linspace(1e-4, 1, 100);
    Gain = p ./ sqrt(1 + r.^a);
    plot(p, Gain, '-', 'LineWidth', 1.5);
    Legend_a{a}=strcat('a=', num2str(a));
    hold on 
end
legend(Legend_a, 'Location','Best');
hold off;

%% Under this scheme, how much does the ranked probability pr change?
% Calculate effects of increasing rank on ranked probability.
% Calculate new curve for each new value of r.
gamma = 2;
a=2;
figure;
xlabel('Probability of cytoplasm edge (p)');    
ylabel('Ranked probability (p_r)');      
title('Effect of increasing edge confidence and rank on ranked probability (a=3, gamma=2)'); 
hold on            
i=0;
RL = zeros(6, 100);
for r = 1: 6
    p = linspace(1e-4, 1, 100);
    Gain = 1 / sqrt(1 + r^a);
    pr = p .* Gain;    
    RL(i+1,:) = - (1 - pr).^gamma .* log(pr);
    plot(p, pr, 'LineWidth', 1.5);
    i = i+1;                        % increase the index by one
    Legend_r{i}=strcat('r=', num2str(i)); % store the current value of Ao for displaying it on the legend
    hold on 
end
legend(Legend_r, 'Location','NorthWest');
hold off;

%% Comparison between CE, FL, and RL  
gamma = 2;
a=2;
p = linspace(1e-4, 1, 100);
CE = -log(p);
FL = -(1-p).^gamma .* log(p);
figure;
% plot(p, CE, '-o');
hold on;
% plot(p, FL, '-*');
xlabel('Probability of cytoplasm edge (p)');    
ylabel('Loss'); 
% Legend_r{1} = 'CE (\gamma =0)';
% Legend_r{2} = 'FL (r=0)';
i=1;
for r = 0: 6
    Gain = 1 / sqrt(1 + r^a);
    pr = p .* Gain;    
    RL = -0.5.*(pr).^gamma .* log(1-pr);
    plot(p, RL, 'LineWidth', 1.5);
    Legend_r{i} = strcat('r=', num2str(r)); 
    hold on 
    i = i+1;                      
end
legend(Legend_r, 'Location','NorthEast');
hold off;


%% 1. Effect of increasing confidence on loss
gamma = 2;

% Calculate relation between increasing p and ranking loss
% Calculate new curve for each new value of p.
figure(1);
xlabel('Rank');           % add a label to the x axis
ylabel('Loss');       % add a label to the y axis
hold on             % hold on the figure to plot more on top
i = 0;              % variable to use an matrix index in error_Ao below for storing the error values at each iteration
for p_out = 0.1: 0.1: 1
    i = linspace(1, 5);
    pr = p_out ./ (exp(i));  
    RL = - (1 - pr).^gamma .* log(pr);
    plot(i, RL);
    i = i+1;                        % increase the index by one
    Legend_p{i}=strcat('p=', num2str(p_out)); % store the current value of Ao for displaying it on the legend
    hold on 
end
legend(Legend_p, 'Location','Best');
hold off;

%% 2. Effect of increasing r on loss
% Calculate relation between increasing p and ranking loss
% Calculate new curve for each new value of p.
gamma = 2;
figure(2);
xlabel('Probability of cytoplasm edge');           % add a label to the x axis
ylabel('Loss');       % add a label to the y axis
title('Effect of increasing rank on loss: p_r = p / e^{|1-r|}');  % add a title to the subplot
hold on             % hold on the figure to plot more on top
i = 0;              % variable to use an matrix index in error_Ao below for storing the error values at each iteration
for i = 1: 1: 5
    p_out = linspace(1e-4, 1, 100);
    pr = p_out ./ exp(abs(1-i));    
    RL = - (1 - pr).^gamma .* log(pr);
    plot(p_out, RL, 'LineWidth', 1.5);
    i = i+1;                        % increase the index by one
    Legend_r{i}=strcat('r=', num2str(i)); % store the current value of Ao for displaying it on the legend
    hold on 
end
legend(Legend_r, 'Location','NorthEast');
hold off;

%% 3. Effect of rank on probability
% As the rank increases, what effect does this have on the probability?
gamma = 2;
figure(3);
xlabel('Sigmoid Output');           % add a label to the x axis
ylabel('Ranked Probability');       % add a label to the y axis
title('Effect of increasing rank on probability: p_r = p / e^{|1-r|}');  % add a title to the subplot
hold on             % hold on the figure to plot more on top
i = 0;              % variable to use an matrix index in error_Ao below for storing the error values at each iteration
for i = 1: 1: 5
    p_out = linspace(1e-4, 1, 100);
    pr = p_out ./ exp(abs(1-i));    
    plot(p_out, pr, 'LineWidth', 1.5);
    i = i+1;                        % increase the index by one
    Legend_r{i}=strcat('r=', num2str(i)); % store the current value of Ao for displaying it on the legend
    hold on 
end
legend(Legend_r, 'Location','NorthEast');
hold off;


%% Individual tries: 
% Changing difficulty, r
% Try: Pr = p / r
% When r = 1, we have the vanilla focal loss:
%FL = - (1 - p)^gamma * log(p);

% Doing this iteratively for increasing r and keeping p constant (0.2):
% Investigate ranking loss (RL):
gamma = 2; 
i = 1;
p_out = 0.1;
pr = p_out ./ exp(abs(1-i));    
RL = - (1 - pr).^gamma .* log(pr);
% figure(1);
% plot(r, RL);
% title('Ranking loss for increasing difficulty r (p = 0.2)');
% xlabel('Rank');
% ylabel('Loss');

%% Changing confidence
% This time see what happens when the network gets more and more confident
% that its found an edge pixel. Keep r constant.
p_out = linspace(1e-4, 1);
i = 2;
pr = p_out ./(exp(i));    
RL = - (1 - pr).^gamma .* log(pr);
figure(2);
plot(p_out, RL);
title('RL for increasing confidence p (r = 2)');
xlabel('Probability of cytoplasm edge');
ylabel('Loss');

