function [afterSplitImg, afterMergeImg, mergedLabelMap] = split_and_merge(img, T_split, T_merge, minSize)
    if nargin < 2 || isempty(T_split), T_split = 15; end
    if nargin < 3 || isempty(T_merge), T_merge = 25; end
    if nargin < 4 || isempty(minSize), minSize = 4; end

    imgDouble = double(img);
    [H, W, C] = size(imgDouble);

    splitLabelMap = zeros(H, W);
    numRegions = 0;
    maxEstRegions = ceil((H * W) / max(1, minSize^2)) * 2;
    regCount = zeros(maxEstRegions, 1);
    regSum   = zeros(maxEstRegions, C);

    quadSplit(1, H, 1, W);

    regCount = regCount(1:numRegions);
    regSum   = regSum(1:numRegions, :);

    leftL  = splitLabelMap(:, 1:end-1);
    rightL = splitLabelMap(:, 2:end);
    pairsH = [leftL(leftL ~= rightL), rightL(leftL ~= rightL)];

    topL   = splitLabelMap(1:end-1, :);
    botL   = splitLabelMap(2:end, :);
    pairsV = [topL(topL ~= botL), botL(topL ~= botL)];

    adjPairs = unique([pairsH; pairsV], 'rows');

    parent = transpose(1:numRegions);

    function root = findRoot(node)
        while parent(node) ~= node
            parent(node) = parent(parent(node));
            node = parent(node);
        end
        root = node;
    end

    for idx = 1:size(adjPairs, 1)
        u = findRoot(adjPairs(idx, 1));
        v = findRoot(adjPairs(idx, 2));

        if u ~= v
            meanU = regSum(u, :) / regCount(u);
            meanV = regSum(v, :) / regCount(v);

            colorDiff = norm(meanU - meanV) / sqrt(C);

            if colorDiff <= T_merge
                parent(v) = u;
                regSum(u, :) = regSum(u, :) + regSum(v, :);
                regCount(u)  = regCount(u) + regCount(v);
            end
        end
    end

    rootList = zeros(numRegions, 1);
    for k = 1:numRegions
        rootList(k) = findRoot(k);
    end
    [~, ~, newLabels] = unique(rootList);
    mergedLabelMap = reshape(newLabels(splitLabelMap), H, W);

    rng(42);
    afterSplitImg = label2rgb(splitLabelMap, 'jet', 'k', 'shuffle');
    rng(42);
    afterMergeImg = label2rgb(mergedLabelMap, 'jet', 'k', 'shuffle');

    function quadSplit(r1, r2, c1, c2)
        h = r2 - r1 + 1;
        w = c2 - c1 + 1;
        if h <= 0 || w <= 0, return; end

        patch = imgDouble(r1:r2, c1:c2, :);
        pixels = reshape(patch, [], C);

        stdVal = mean(std(pixels, 0, 1));

        if (stdVal <= T_split) || (h <= minSize && w <= minSize)
            numRegions = numRegions + 1;
            splitLabelMap(r1:r2, c1:c2) = numRegions;
            regCount(numRegions) = h * w;
            regSum(numRegions, :) = sum(pixels, 1);
            return;
        end

        rMid = floor((r1 + r2) / 2);
        cMid = floor((c1 + c2) / 2);

        if h > minSize && w > minSize
            quadSplit(r1, rMid, c1, cMid);
            quadSplit(r1, rMid, cMid + 1, c2);
            quadSplit(rMid + 1, r2, c1, cMid);
            quadSplit(rMid + 1, r2, cMid + 1, c2);
        elseif h > minSize
            quadSplit(r1, rMid, c1, c2);
            quadSplit(rMid + 1, r2, c1, c2);
        else
            quadSplit(r1, r2, c1, cMid);
            quadSplit(r1, r2, cMid + 1, c2);
        end
    end
end