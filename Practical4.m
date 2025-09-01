img = imread('C:\Users\mithi\OneDrive\ドキュメント\MATLAB\MATLAB\dramatic-scenery-3840x2160-20802.jpg');

if size(img, 3) == 3
    grayImg = rgb2gray(img);
else
    grayImg = img;
end

keypoints = detectSIFTFeatures(grayImg);
[features, valid_points] = extractFeatures(grayImg, keypoints);

imshow(img);
hold on;
plot(valid_points.selectStrongest(375));
title('Detected SIFT Keypoints');
hold off;
