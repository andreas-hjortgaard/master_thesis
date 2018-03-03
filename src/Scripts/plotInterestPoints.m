function [iPoints, X, Y, C] = plotInterestPoints(imageFile, interestPointFile)
% interestPoints(imageFile, interestPointFile)
%
% Takes the interest points from a binary file and plots them on the image
% file. Also makes a plot of the visual words.
% Saves the figures on disk.

% load interest points from binary file
fileID = fopen(interestPointFile,'rb');
binData = fread(fileID, inf, 'ushort');
numPoints = length(binData)/3;

X = binData(1:numPoints);
Y = binData(numPoints+1:2*numPoints);
C = binData(2*numPoints+1:end);
C = C + 1; % convert from zero-indexed to one-indexed

% load image
rgbImg = imread(imageFile);
grayImg = rgb2gray(rgbImg);
[n m] = size(grayImg);

% put interest points in a matrix - NOT needed for plotting
iPoints = zeros(n,m);
for i=1:numPoints
    iPoints(Y(i),X(i)) = C(i);
end

% plot interest points on grayscale image
figure;
imshow(grayImg);
hold on;
plot(X,Y,'.b');
hold off;

% save figure
[pathstr, name, ext] = fileparts(imageFile);
iPointsFile = fullfile(pathstr, [name, '_interest_points.eps']);
print(iPointsFile,'-depsc2');

% plot visual words on grayscale image
figure;
imshow(grayImg);
cc = hsv(3000); % create 3000 different colors from the HSV colormap
hold on;
for i=1:numPoints
  plot(X(i),Y(i),'.','color',cc(C(i),:));
end
hold off;

% save figure
visualWordsFile = fullfile(pathstr, [name, '_visual_words.eps']);
print(visualWordsFile,'-depsc2');

end
