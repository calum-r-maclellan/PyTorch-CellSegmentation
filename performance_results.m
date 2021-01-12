% Wee script to read the performance results .txt files and get mean(std)

clear; close all; clc;
% root = 'E:\ISBI_data_code\Quantitative_results\loss_functions\Dice+Beta\tnr_log\tnr_nuclei.txt';
root = '/Volumes/CALUM_ENGD/ISBI_CellSegmentation/CytoDetector_results//tnr_log/tnr_nuclei.txt';
file = fopen(root, 'r'); % open file for reading
formatSpec = '%f';       % specify data format

% Read file
results = fscanf(file, formatSpec);

% Compute mean(std) 
avg = mean(results);
std = std(results);

fclose(file);
