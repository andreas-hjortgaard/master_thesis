function plotContours(imageFile, marginalPath, category, partition, imageNumber, weightMethod, stepSize, width, height, marginalMethod)
%
% plotContours(imageFile, marginalPath, category, partition, imageNumber, method, stepSize, width, height, marginalMethod)

if (nargin < 10)
    marginalMethod = '';
end

% load data
rgbImage = imread(imageFile);
cornerLT = load(fullfile(marginalPath, [category,'_',partition,'_imageNumber_',num2str(imageNumber),'_',weightMethod,'_stepSize_',num2str(stepSize),'_width_',num2str(width),'_height_',num2str(height),'_',marginalMethod,'LT.txt']));
cornerLB = load(fullfile(marginalPath, [category,'_',partition,'_imageNumber_',num2str(imageNumber),'_',weightMethod,'_stepSize_',num2str(stepSize),'_width_',num2str(width),'_height_',num2str(height),'_',marginalMethod,'LB.txt']));
cornerRT = load(fullfile(marginalPath, [category,'_',partition,'_imageNumber_',num2str(imageNumber),'_',weightMethod,'_stepSize_',num2str(stepSize),'_width_',num2str(width),'_height_',num2str(height),'_',marginalMethod,'RT.txt']));
cornerRB = load(fullfile(marginalPath, [category,'_',partition,'_imageNumber_',num2str(imageNumber),'_',weightMethod,'_stepSize_',num2str(stepSize),'_width_',num2str(width),'_height_',num2str(height),'_',marginalMethod,'RB.txt']));

% plot image in color
figure;
imshow(rgbImage);

% plot image in grayscale
grayImage = rgb2gray(rgbImage);
figure;
imshow(grayImage); 

% marginals: reshape to image dimension
marginalLT = transpose(reshape(cornerLT, width, height));
marginalLB = transpose(reshape(cornerLB, width, height));
marginalRT = transpose(reshape(cornerRT, width, height));
marginalRB = transpose(reshape(cornerRB, width, height));


% normalize to range [0,1]
%marginalLT = (marginalLT - min(marginalLT(:))) / (max(marginalLT(:))-min(marginalLT(:)));
%marginalLB = (marginalLB - min(marginalLB(:))) / (max(marginalLB(:))-min(marginalLB(:)));
%marginalRT = (marginalRT - min(marginalRT(:))) / (max(marginalRT(:))-min(marginalRT(:)));
%marginalRB = (marginalRB - min(marginalRB(:))) / (max(marginalRB(:))-min(marginalRB(:)));

% plot marginals - only visible if normalized
%figure;
%subplot(2,2,1);
%subimage(marginalLT);
%subplot(2,2,2);
%subimage(marginalRT);
%subplot(2,2,3);
%subimage(marginalLB);
%subplot(2,2,4);
%subimage(marginalRB);


% convert grayscale image back to RGB
rgbGrayImage = repmat(grayImage, [1 1 3]);

% if stepSize is 2 or 4, then interpolation is needed
if (stepSize == 2)
    marginalLT = interp2(marginalLT,1);
    marginalLB = interp2(marginalLB,1);
    marginalRT = interp2(marginalRT,1);
    marginalRB = interp2(marginalRB,1);
elseif (stepSize == 4)
    marginalLT = interp2(marginalLT,2);
    marginalLB = interp2(marginalLB,2);
    marginalRT = interp2(marginalRT,2);
    marginalRB = interp2(marginalRB,2);
end

% plot contour on image
n = 5;
figure;
imshow(rgbGrayImage);
hold on;
contour(marginalLT,n,'-r','LineWidth',2);
contour(marginalLB,n,'-g','LineWidth',2);
contour(marginalRT,n,'-b','LineWidth',2);
contour(marginalRB,n,'-c','LineWidth',2);
hold off;

% save figure
[pathstr, name, ext] = fileparts(imageFile);
contourFile = fullfile(pathstr, [name, '_', marginalMethod, '_contours_', weightMethod, '_stepSize_', num2str(stepSize), '.eps']);
print(contourFile,'-depsc2');

% plot contour on image with labeled contours
figure;
imshow(rgbGrayImage);
hold on;
[cLT,hLT] = contour(marginalLT,n,'-r','LineWidth',2);
[cLB,hLB] = contour(marginalLB,n,'-g','LineWidth',2);
[cRT,hRT] = contour(marginalRT,n,'-b','LineWidth',2);
[cRB,hRB] = contour(marginalRB,n,'-c','LineWidth',2);
clabel(cLT,hLT);
clabel(cLB,hLB);
clabel(cRT,hRT);
clabel(cRB,hRB);
hold off;

% save figure
contourLabeledFile = fullfile(pathstr, [name, '_', marginalMethod, '_contours_', weightMethod, '_stepSize_', num2str(stepSize), '_labeled.eps']);
print(contourLabeledFile,'-depsc2');

end
