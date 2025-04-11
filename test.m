close all; clear; clc;

% Retrieve file names and sex labels from data file
sheetFileName = 'BVC_Voice_Bio_Public.xlsx';
data = readtable(sheetFileName, 'Sheet', 'voice_bio_data', 'Range', 'A:B');
% audioFolderPath = 'S_02_voice/S_02/multiple_sentences/';
audioFolderPath = 'multiple_sentences/multiple_sentences/';
audioFiles = dir(fullfile(audioFolderPath, '*.wav'));

file = fullfile(audioFolderPath, audioFiles(1).name);

[audioData, fs] = audioread(file);

coeffs = mfcc(audioData, fs);
meanMFCC = mean(coeffs);
varMFCC = var(coeffs);

centroid = spectralCentroid(audioData, fs);
meanCen = mean(centroid);
varCen = var(centroid);


features = [];
