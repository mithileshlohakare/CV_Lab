clear;
clc;
close all;

original_image_rgb = imread('pp.jpg');
original_image_gray = rgb2gray(original_image_rgb);
original_image_double = double(original_image_gray);

averaging_filter = ones(3,3) / 9;
smoothed_image_avg = conv2(original_image_double, averaging_filter, 'same');

gaussian_filter = fspecial('gaussian', [5 5], 8);
smoothed_image_gaussian = conv2(original_image_double, gaussian_filter, 'same');

smoothed_image_avg_norm = mat2gray(smoothed_image_avg);
smoothed_image_gaussian_norm = mat2gray(smoothed_image_gaussian);

figure;
subplot(1,3,1), imshow(original_image_gray), title('Original Grayscale');
subplot(1,3,2), imshow(smoothed_image_avg_norm), title('Averaging Filter');
subplot(1,3,3), imshow(smoothed_image_gaussian_norm), title('Gaussian Filter');
