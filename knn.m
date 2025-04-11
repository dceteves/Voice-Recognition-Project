close all; clear; clc;

% Retrieve file names and sex labels from data file
sheetFileName = 'BVC_Voice_Bio_Public.xlsx';
data = readtable(sheetFileName, 'Sheet', 'voice_bio_data', 'Range', 'A:B');
audioFolderPath = 'multiple_sentences/multiple_sentences/';
audioFiles = dir(fullfile(audioFolderPath, '*.wav'));

% Initialize X and y tables
n = length(audioFiles);
nfeatures = 56;
sampleRate = 48000;
% fileLength = 3; % in seconds
% fileLengthInSamples = sampleRate * fileLength;

% Initialize X and y vectors
X = zeros(n, nfeatures);
y = zeros(n, 1);

% Loop through files
for i = 1:n

    % Get the file name
    audioFileName = audioFiles(i).name;
    fullFileName = fullfile(audioFolderPath, audioFileName);
   
    % Read audio file 
    [audioData, fs] = audioread(fullFileName);

    % Pad the audio data for consistent sample sizing
    % audioData = audioData(1);
    % numSamples = length(audioData);
    % if numSamples < fileLengthInSamples
    %     padded = [audioData; zeros(fileLengthInSamples - numSamples, 1)];
    % else
    %     padded = audioData;
    % end

    % Retrieve label
    id = str2double(audioFileName(6:9));
    sex = data(ismember(data.New_ID, id), :).Sex{1};
    if sex == "'Male'"
        sex_label = 0;
    elseif sex == "'Female'"
        sex_label = 1;
    end

    % Append feature & label to vector
    features = extract_features(audioData, fs);
    for j = 1:length(features)
        X(i, j) = features(j);
    end
    y(i) = sex_label;
end

% Data Balancing
majorityClass = mode(y);
minorityCount = min(histcounts(y));
keep = false(size(y));
for class = unique(y)'
    classInd = find(y == class);
    if class == majorityClass
        selected = randperm(length(classInd), minorityCount);
        keep(classInd(selected)) = true;
    else
        keep(classInd) = true;
    end
end
X = X(keep, :);
y = y(keep);

% Normalize data
Xmin = min(X(:));
Xmax = max(X(:));
X = (X - Xmin) / (Xmax - Xmin);

% Split into train/test sets
cv = cvpartition(y, 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);
Xtrain = X(idxTrain, :);
ytrain = y(idxTrain);
Xtest = X(idxTest, :);
ytest = y(idxTest);

% Standardize features
% Xtrain = (Xtrain - mean(Xtrain, 1)) ./ std(Xtrain, 0, 1);
% Xtest = (Xtest - mean(Xtrain, 1)) ./ std(Xtrain, 0, 1);

% Fit into KNN model; test each k value up to 50 
accs_1 = [];
accs_2 = [];
for k = 1:50

    % Built in KNN classifier
    knnModel = fitcknn(Xtrain, ytrain, 'NumNeighbors', k);
    ypred_1 = predict(knnModel, Xtest);

    % Implemented KNN classifier
    % ypred_1 = KNN(Xtrain, ytrain, Xtest, k);
    ypred_2 = findKNN(Xtrain, ytrain, Xtest, k);
    
    % Calculate the accuracy
    acc_1 = sum(ypred_1 == ytest) / length(ytest);
    acc_2 = sum(ypred_2 == ytest) / length(ytest);

    accs_1 = [accs_1; acc_1];
    accs_2 = [accs_2; acc_2];

    % Confusion Matrix
    cm_1 = confusionmat(ytest, ypred_1);
    cm_2 = confusionmat(ytest, ypred_2);
    
    % Display & compare metrics
    fprintf('k = %d\n', k);
    fprintf('MATLAB KNN Classification Accuracy: %.2f%%\n', acc_1 * 100);
    disp('MATLAB KNN Confusion Matrix:');
    disp(cm_1);
    fprintf('Custom KNN Classification Accuracy: %.2f%%\n', acc_2 * 100);
    disp('Custom KNN Confusion Matrix:');
    disp(cm_2);
end

[max_1, ind_1] = max(accs_1(:));
[max_2, ind_2] = max(accs_2(:));

fprintf('Best accuracy for MATLAB KNN: %.2f% @ k = %d%\n', max_1 * 100, ind_1);
fprintf('Best accuracy for Custom KNN: %.2f% @ k = %d%\n', max_2 * 100, ind_2);

function features = extract_features(audio, fs)
    coeffs = mfcc(audio, fs);
    meanMFCC = mean(coeffs);
    varMFCC = var(coeffs);
    features = [meanMFCC varMFCC];
end

function ypred = findKNN(Xtrain, ytrain, Xtest, k)
    ntrain = size(Xtrain, 1);
    ntest = size(Xtest, 1);
    dists = zeros(ntest, ntrain);

    for i = 1:ntest
        for j = 1:ntrain
            dists(i, j) = sqrt(sum(Xtest(i, :) - Xtrain(j, :)) .^ 2);
        end
    end
    
    [sortedDists, sortedInd] = sort(dists, 2, 'ascend');
    indices = sortedInd(:, 1:k);

    ypred = zeros(ntest, 1);

    for i = 1:ntest
        nearest = ytrain(indices(i, :));
        ypred(i) = mode(nearest);
    end
end
