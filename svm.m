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
X = zeros(n, nfeatures);
y = zeros(n, 1);

% Loop through files
for i = 1:n

    % Get the file name
    audioFileName = audioFiles(i).name;
    fullFileName = fullfile(audioFolderPath, audioFileName);
   
    % Read audio file 
    [audioData, fs] = audioread(fullFileName);

    % Low-pass filter for noise reduction
    audioData = clean(audioData, fs);
   
    % % Pad the audio data for consistent sample sizing
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

Xmin = min(X(:));
Xmax = max(X(:));
X = (X - Xmin) / (Xmax - Xmin);

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

svmModel = fitcsvm(Xtrain, ytrain, ...
    'KernelFunction', 'linear', ...
    'Standardize', true, ...
    'ClassNames', unique(y));

ypred = predict(svmModel, Xtest);

% Calculate the accuracy
accuracy = sum(ypred == ytest) / length(ytest);

% Display the accuracy
fprintf('Classification Accuracy: %.2f%%\n', accuracy * 100);

% Confusion Matrix
confusionMat = confusionmat(ytest, ypred);
disp('Confusion Matrix:');
disp(confusionMat);

function features = extract_features(audio, fs)
    coeffs = mfcc(audio, fs);
    meanMFCC = mean(coeffs);
    varMFCC = var(coeffs);
    features = [meanMFCC varMFCC];
end

function audioOut = clean(audioData, fs)    
    fc = 5000;
    [b, a] = butter(6, fc/(fs/2));
    audioOut = filter(b, a, audioData);
end