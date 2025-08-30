img = imread('C:\Users\mithi\OneDrive\ドキュメント\MATLAB\MATLAB\dramatic-scenery-3840x2160-20802.jpg');

if size(img, 3) == 3
    grayImg = rgb2gray(img);
else
    grayImg = img;
end

[hogFeatures, hogVisualization] = extractHOGFeatures(grayImg, ...
    'CellSize', [128 128], 'NumBins', 64);

figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(img);
hold on;
plot(hogVisualization);
title('HOG Features Visualization');

fprintf('Size of HOG feature vector: %d\n', numel(hogFeatures));
