function [edgeBinary, objectMask, segmentedObj] = edge_segmentation(img, operatorName, edgeThresh, morphRadius)
    if nargin < 3 || isempty(edgeThresh), edgeThresh = 0.15; end
    if nargin < 4 || isempty(morphRadius), morphRadius = 5; end

    if size(img, 3) == 3
        gray = double(rgb2gray(img));
    else
        gray = double(img);
    end

    operatorName = lower(strtrim(operatorName));

    switch operatorName
        case 'sobel'
            Gx = [-1 0 1; -2 0 2; -1 0 1];
            Gy = [-1 -2 -1; 0 0 0; 1 2 1];
            Ix = conv2(gray, Gx, 'same');
            Iy = conv2(gray, Gy, 'same');
            edgeMag = sqrt(Ix.^2 + Iy.^2);

        case 'prewitt'
            Gx = [-1 0 1; -1 0 1; -1 0 1];
            Gy = [-1 -1 -1; 0 0 0; 1 1 1];
            Ix = conv2(gray, Gx, 'same');
            Iy = conv2(gray, Gy, 'same');
            edgeMag = sqrt(Ix.^2 + Iy.^2);

        case 'roberts'
            Gx = [1 0; 0 -1];
            Gy = [0 1; -1 0];
            Ix = conv2(gray, Gx, 'same');
            Iy = conv2(gray, Gy, 'same');
            edgeMag = sqrt(Ix.^2 + Iy.^2);

        case 'laplace'
            K_lap = [1 1 1; 1 -8 1; 1 1 1];
            edgeMag = abs(conv2(gray, K_lap, 'same'));

        case 'log'
            sigma = 1.4;
            halfSize = ceil(3 * sigma);
            [x, y] = meshgrid(-halfSize:halfSize, -halfSize:halfSize);
            G = exp(-(x.^2 + y.^2) / (2 * sigma^2));
            G = G / sum(G(:)); % Normalisasi kernel Gaussian
            
            smoothed = conv2(gray, G, 'same');
            K_lap = [0 1 0; 1 -4 1; 0 1 0];
            edgeMag = abs(conv2(smoothed, K_lap, 'same'));

        case 'canny'
            if isscalar(edgeThresh)
                edgeBinary = edge(uint8(gray), 'canny');
            else
                edgeBinary = edge(uint8(gray), 'canny', edgeThresh);
            end

        otherwise
            error('Operator tidak dikenali. Pilih: sobel, prewitt, roberts, laplace, log, atau canny.');
    end

    if ~strcmp(operatorName, 'canny')
        maxVal = max(edgeMag(:));
        if maxVal > 0
            edgeMag = edgeMag / maxVal;
        end
        edgeBinary = edgeMag > edgeThresh;
    end

    se = strel('disk', morphRadius);
    closedEdges = imclose(edgeBinary, se);

    filledMask = imfill(closedEdges, 'holes');

    minObjArea = round(0.01 * numel(gray));
    objectMask = bwareaopen(filledMask, minObjArea);

    if ~any(objectMask(:)) && any(filledMask(:))
        cc = bwconncomp(filledMask);
        numPixels = cellfun(@numel, cc.PixelIdxList);
        [~, idxMax] = max(numPixels);
        objectMask = false(size(gray));
        objectMask(cc.PixelIdxList{idxMax}) = true;
    end

    segmentedObj = bsxfun(@times, uint8(img), uint8(objectMask));
end