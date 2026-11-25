function gui
functionsDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(fileparts(functionsDir));
pythonScript = fullfile(functionsDir, 'predict.py');
checkpointPath = fullfile(projectDir, 'models', 'camvid_deeplabv3.pth');
addpath(functionsDir);

inputPath = '';
resultImage = [];

fig = uifigure('Name', 'Segmentasi Citra', ...
    'Position', [100 50 1400 900]);
controlPanel = uipanel(fig, 'Title', 'Kontrol', ...
    'Position', [10 10 310 870]);

uibutton(controlPanel, 'Text', 'Pilih gambar...', ...
    'Position', [20 810 270 34], 'ButtonPushedFcn', @chooseImage);
inputLabel = uilabel(controlPanel, 'Text', 'Belum ada gambar dipilih', ...
    'Position', [20 760 270 42], 'WordWrap', 'on');

uilabel(controlPanel, 'Text', 'Metode', ...
    'Position', [20 725 270 22]);
methodField = uidropdown(controlPanel, ...
    'Items', {'Deep learning', 'Deteksi tepi', 'Split and merge'}, ...
    'Position', [20 695 270 30]);

uilabel(controlPanel, 'Text', 'Checkpoint model (.pth)', ...
    'Position', [20 660 270 22]);
checkpointField = uieditfield(controlPanel, 'text', ...
    'Value', checkpointPath, 'Position', [20 630 215 28]);
uibutton(controlPanel, 'Text', '...', ...
    'Position', [243 630 47 28], 'ButtonPushedFcn', @chooseCheckpoint);

uilabel(controlPanel, 'Text', 'Python executable', ...
    'Position', [20 595 270 22]);
pythonField = uieditfield(controlPanel, 'text', ...
    'Value', 'python', 'Position', [20 565 270 28]);

uilabel(controlPanel, 'Text', 'Operator deteksi tepi', ...
    'Position', [20 530 270 22]);
edgeOperatorField = uidropdown(controlPanel, ...
    'Items', {'canny', 'sobel', 'prewitt', 'roberts', 'laplace', 'log'}, ...
    'Position', [20 500 270 28]);

uilabel(controlPanel, 'Text', 'Ambang tepi (0-1; Canny otomatis)', ...
    'Position', [20 465 270 22]);
edgeThresholdField = uieditfield(controlPanel, 'numeric', ...
    'Value', 0.15, 'Limits', [0 1], 'Position', [20 435 270 28]);

uilabel(controlPanel, 'Text', 'Radius morfologi tepi', ...
    'Position', [20 400 270 22]);
morphRadiusField = uieditfield(controlPanel, 'numeric', ...
    'Value', 5, 'Limits', [0 30], 'Position', [20 370 270 28]);

uilabel(controlPanel, 'Text', 'T split (0-255)', ...
    'Position', [20 335 270 22]);
splitThresholdField = uieditfield(controlPanel, 'numeric', ...
    'Value', 15, 'Limits', [0 255], 'Position', [20 305 270 28]);

uilabel(controlPanel, 'Text', 'T merge (0-255)', ...
    'Position', [20 270 270 22]);
mergeThresholdField = uieditfield(controlPanel, 'numeric', ...
    'Value', 25, 'Limits', [0 255], 'Position', [20 240 270 28]);

uilabel(controlPanel, 'Text', 'Ukuran blok minimum (piksel)', ...
    'Position', [20 205 270 22]);
blockSizeField = uieditfield(controlPanel, 'numeric', ...
    'Value', 4, 'Limits', [1 128], 'Position', [20 175 270 28]);

uibutton(controlPanel, 'Text', 'Proses gambar', ...
    'Position', [20 120 270 38], 'ButtonPushedFcn', @processImage);
saveButton = uibutton(controlPanel, 'Text', 'Simpan hasil...', ...
    'Position', [20 75 270 34], 'Enable', 'off', ...
    'ButtonPushedFcn', @saveResult);
statusLabel = uilabel(controlPanel, 'Text', 'Siap.', ...
    'Position', [20 15 270 50], 'WordWrap', 'on', ...
    'VerticalAlignment', 'top');

inputAxes = uiaxes(fig, 'Position', [345 70 500 760]);
title(inputAxes, 'Gambar input');
inputAxes.XTick = [];
inputAxes.YTick = [];
inputAxes.Box = 'on';
resultAxes = uiaxes(fig, 'Position', [875 70 500 760]);
title(resultAxes, 'Hasil');
resultAxes.XTick = [];
resultAxes.YTick = [];
resultAxes.Box = 'on';
edgeSegmentAxes = uiaxes(fig, 'Position', [1035 70 340 760], ...
    'Visible', 'off');
title(edgeSegmentAxes, 'Hasil segmentasi deteksi tepi');
edgeSegmentAxes.XTick = [];
edgeSegmentAxes.YTick = [];
edgeSegmentAxes.Box = 'on';

    function chooseImage(~, ~)
        [file, folder] = uigetfile( ...
            {'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff', 'File gambar'}, ...
            'Pilih gambar');
        if isequal(file, 0)
            return;
        end
        inputPath = fullfile(folder, file);
        inputLabel.Text = file;
        imshow(readImageRGB(inputPath), 'Parent', inputAxes);
        title(inputAxes, 'Gambar input');
        inputAxes.Position = [345 70 500 760];
        resultAxes.Position = [875 70 500 760];
        cla(resultAxes);
        title(resultAxes, 'Hasil');
        cla(edgeSegmentAxes);
        edgeSegmentAxes.Visible = 'off';
        resultImage = [];
        saveButton.Enable = 'off';
        statusLabel.Text = 'Gambar berhasil dimuat.';
    end

    function chooseCheckpoint(~, ~)
        [file, folder] = uigetfile('*.pth', 'Pilih checkpoint model');
        if isequal(file, 0)
            return;
        end
        checkpointField.Value = fullfile(folder, file);
    end

    function processImage(~, ~)
        if isempty(inputPath)
            uialert(fig, 'Pilih gambar input terlebih dahulu.', 'Input diperlukan');
            return;
        end

        image = readImageRGB(inputPath);
        method = methodField.Value;
        statusLabel.Text = 'Sedang memproses gambar...';
        drawnow;

        try
            segmentedResult = [];
            switch method
                case 'Deep learning'
                    result = runDeepLearning();
                case 'Deteksi tepi'
                    [result, segmentedResult] = runEdgeDetection(image);
                case 'Split and merge'
                    result = runSplitAndMerge(image);
                otherwise
                    error('Metode pemrosesan tidak dikenali.');
            end
            resultImage = result;
            if strcmp(method, 'Deteksi tepi')
                inputAxes.Position = [345 70 330 760];
                resultAxes.Position = [690 70 330 760];
                edgeSegmentAxes.Position = [1035 70 340 760];
                edgeSegmentAxes.Visible = 'on';
                imshow(segmentedResult, 'Parent', edgeSegmentAxes);
            else
                inputAxes.Position = [345 70 500 760];
                resultAxes.Position = [875 70 500 760];
                edgeSegmentAxes.Visible = 'off';
                cla(edgeSegmentAxes);
            end
            imshow(resultImage, 'Parent', resultAxes);
            title(resultAxes, method);
            statusLabel.Text = 'Pemrosesan selesai.';
            saveButton.Enable = 'on';
        catch exception
            statusLabel.Text = ['Gagal: ' exception.message];
            uialert(fig, exception.message, 'Pemrosesan gagal');
        end
    end

    function result = runDeepLearning()
        checkpoint = strtrim(checkpointField.Value);
        pythonExe = strtrim(pythonField.Value);
        if isempty(checkpoint) || ~isfile(checkpoint)
            error('Checkpoint tidak ditemukan. Pilih file best_model.pth yang sudah dilatih.');
        end
        if isempty(pythonExe)
            error('Isi lokasi executable Python.');
        end
        if ~isfile(pythonScript)
            error('Skrip inferensi tidak ditemukan: %s', pythonScript);
        end

        maskPath = [tempname '.png'];
        overlayPath = [tempname '.png'];
        cleanup = onCleanup(@() removeTemporaryFiles(maskPath, overlayPath));
        command = strjoin({quoteArgument(pythonExe), ...
            quoteArgument(pythonScript), ...
            '--image', quoteArgument(inputPath), ...
            '--checkpoint', quoteArgument(checkpoint), ...
            '--mask-output', quoteArgument(maskPath), ...
            '--overlay-output', quoteArgument(overlayPath), ...
            '--device', 'auto'}, ' ');
        [exitCode, output] = system(command);
        if exitCode ~= 0
            error('Inferensi Python gagal:\n%s', strtrim(output));
        end
        if ~isfile(overlayPath)
            error('Skrip Python selesai tanpa menghasilkan gambar prediksi.');
        end
        result = imread(overlayPath);
    end

    function [result, segmentedResult] = runEdgeDetection(image)
        [edgeMap, ~, segmentedResult] = edge_segmentation(image, ...
            edgeOperatorField.Value, edgeThresholdField.Value, ...
            round(morphRadiusField.Value));
        result = im2double(image);
        result(:, :, 1) = max(result(:, :, 1), double(edgeMap));
        result(:, :, 2) = result(:, :, 2) .* ~edgeMap;
        result(:, :, 3) = result(:, :, 3) .* ~edgeMap;
    end

    function result = runSplitAndMerge(image)
        originalSize = size(image, [1 2]);
        scale = min(1, 640 / max(originalSize));
        workingImage = image;
        if scale < 1
            workingImage = imresize(workingImage, scale);
        end
        [~, result, ~] = split_and_merge(workingImage, ...
            splitThresholdField.Value, mergeThresholdField.Value, ...
            round(blockSizeField.Value));
        if scale < 1
            result = imresize(result, originalSize, 'nearest');
        end
    end

    function saveResult(~, ~)
        if isempty(resultImage)
            return;
        end
        [file, folder] = uiputfile( ...
            {'*.png', 'PNG image'; '*.jpg', 'JPEG image'}, ...
            'Simpan hasil', 'hasil_segmentasi.png');
        if isequal(file, 0)
            return;
        end
        imwrite(resultImage, fullfile(folder, file));
        statusLabel.Text = ['Hasil disimpan: ' fullfile(folder, file)];
    end
end

function image = readImageRGB(path)
[image, colorMap] = imread(path);
if ~isempty(colorMap)
    image = im2uint8(ind2rgb(image, colorMap));
elseif ismatrix(image)
    image = repmat(image, 1, 1, 3);
elseif size(image, 3) > 3
    image = image(:, :, 1:3);
end
end

function quoted = quoteArgument(argument)
argument = char(argument);
quoted = ['"' strrep(argument, '"', '\"') '"'];
end

function removeTemporaryFiles(varargin)
for index = 1:nargin
    if isfile(varargin{index})
        delete(varargin{index});
    end
end
end
