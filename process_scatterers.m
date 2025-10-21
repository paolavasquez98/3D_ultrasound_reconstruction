function process_scatterers(mode)
    % mode can be 'fg', 'mri', 'empty_ellip', 'mesh', 'three_ellip', 'two_chambers'
    % can be used to generate one type of files as:
    % process_scatterers('fg') to process all files in fg mode
    % or to process all of them
    % process_scatterers('all')

    addpath('US_toolbox/Beamforming/');
    addpath('gpuSTA/convolution3D_FFTdomain/');
    addpath('gpuSTA/functions/');
    addpath('MUST/');

    base = 'shape_models';
    all_modes = {'fg', 'mri', 'empty_ellip', 'mesh', 'three_ellip', 'two_chambers'};

    % parse
    if nargin < 1
        error('Usage: process_scatterers(''mode'') or process_scatterers(''mode'', idx)');
    end

    if strcmp(mode, 'all')
        for m = 1:numel(all_modes)
            fprintf('\n Processing ALL: Mode %s \n', all_modes{m});
            process_scatterers(all_modes{m});  % recursive call
        end
        fprintf('\n Finished processing ALL modes.\n');
        return
    end

    switch mode
        case 'fg'
            subdir = 'heart_fg';
            pattern = '*.mat';

        case 'mri'
            subdir = 'mri';
            pattern = '*.nii.gz';

        case 'empty_ellip'
            subdir = 'empty_ellipsoid';
            pattern = '*.mat';

        case 'mesh'
            subdir = 'heart_mesh';
            pattern = '*.mat';

        case 'three_ellip'
            subdir = 'three_ellip';
            pattern = '*.mat';

        case 'two_chambers'
            subdir = 'two_chambers';
            pattern = '*.mat';

        otherwise
            error('Unknown mode: %s', mode);
    end

    scatterer_dir = fullfile(base, subdir, 'scatterers');
    out_fig_dir   = fullfile(base, subdir, 'images');
    out_file_dir  = fullfile(base, subdir, 'beamform');

    % Make sure directories exist
    if ~exist(out_fig_dir, 'dir')
        mkdir(out_fig_dir);
    end
    if ~exist(out_file_dir, 'dir')
        mkdir(out_file_dir);
    end

    % Get files
    scatterer_files = dir(fullfile(scatterer_dir, pattern));
    if isempty(scatterer_files)
        warning('No files found for pattern %s in %s', pattern, scatterer_dir);
        return
    end

    idx_list = 1:numel(scatterer_files);
    fprintf('\n[%s] FOUND %d scatterer files. Processing %d...\n', ...
        mode, numel(scatterer_files), numel(idx_list));

    % Loop through requested files
    for i = idx_list
        scatterer_path = fullfile(scatterer_files(i).folder, scatterer_files(i).name);
        fprintf('\n[%s] Processing file %d/%d: %s\n', mode, i, numel(scatterer_files), scatterer_files(i).name);
        
        try
            process_single(scatterer_path, out_fig_dir, out_file_dir, mode);
        catch ME
            warning('Failed to process %s: %s', scatterer_files(i).name, ME.message);
        end
    end

    fprintf('\n Finished processing mode: %s (%d files)\n', mode, numel(idx_list));
end