clc;
clear;
close all;

img = imread('C:\Users\mithi\OneDrive\ドキュメント\MATLAB\dramatic-scenery-3840x2160-20802.jpg');
img = im2double(img);

figure('Name', 'Original Image');
imshow(img);
title('Original Image');

gray_img = rgb2gray(img);

negative_img = 1 - gray_img;
figure('Name', 'Negative Transformation');
imshow(negative_img);
title('Negative Image');

c = 1; 
log_img = c * log(1 + gray_img);
log_img = (log_img - min(log_img(:))) / (max(log_img(:)) - min(log_img(:)));
figure('Name', 'Logarithmic Transformation');
imshow(log_img);
title('Logarithmic Transformation');

gamma = 0.7; 
c = 1;      
power_law_img = c * (gray_img .^ gamma);
power_law_img = (power_law_img - min(power_law_img(:))) / (max(power_law_img(:)) - min(power_law_img(:)));
figure('Name', 'Power-Law (Gamma) Transformation');
imshow(power_law_img);
title(['Power-Law Transformation, \gamma = ', num2str(gamma)]);
