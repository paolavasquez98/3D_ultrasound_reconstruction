function process_single(scatterer_path, out_fig_dir, out_file_dir, mode)
    % This function processes a single scatterer file and saves the output
    % file and images in the specified directories.
    % scatterer_path: full path to the scatterer file
    % out_fig_dir: directory to save output figures
    % out_file_dir: directory to save output data files

    tic;
    % Reset GPU at the start to avoid leftover memory
    reset(gpuDevice);

    % Load important data files
    load("gpuSTA/matData/irLUT.mat");
    load("gpuSTA/matData/psf_192.mat");
    load('gpuSTA/matData/elementPos.mat');
    nb_element = size(elementPos,1);

    % Load scatterer file
    [~, scatterer_name, ~] = fileparts(scatterer_path);
    fprintf('Processing file: %s\n', scatterer_name);
    
    % Handle different file types
    switch mode 
        case 'fg'
            % Load data
            load(scatterer_path, 'scat_x_fg', 'scat_y_fg', 'scat_z_fg', 'ampl_fg'); 
            scatterers = double([scat_x_fg, scat_y_fg, scat_z_fg]);
            rc_values = ampl_fg;
            amplitudes = randn(size(ampl_fg));   

        case 'mri'
            % Load data
            img = niftiread(scatterer_path);
            % Reorder axis to z,y,x
            img = double(permute(img, [3 2 1]));
            % Preprocess
            img = img - min(img(:));  
            img = img / max(img(:));
            % Get mri sizes
            info = niftiinfo(scatterer_path);
            voxelSize = info.PixelDimensions;
            roidim = (double(size(img)) .* voxelSize([3 2 1])) / 1000;
            % Generate scatterers
            meandist = 0.5e-3;
            [xs, ys, zs, RC] = genscat(roidim, meandist, double(img));
            % Mask scatterers
            [nl, nc, np] = size(img);
            width = roidim(1); height = roidim(2); depth = roidim(3);
            xmin = -width/2; xmax = width/2;
            ymin = -depth/2; ymax = depth/2;
            zmin = 0; zmax = height;
            dxi = (xmax - xmin) / nc;
            dyi = (ymax - ymin) / np;
            dzi = (zmax - zmin) / nl;
            [xi, zi, yi] = meshgrid( ...
                linspace(xmin + dxi/2, xmax - dxi/2, nc), ...
                linspace(zmin + dzi/2, zmax - dzi/2, nl), ...
                linspace(ymin + dyi/2, ymax - dyi/2, np) );
            I_at_scatterers = interp3(xi, zi, yi, img, xs, zs, ys, 'nearest', 0); 
            keep_idx = I_at_scatterers >= 0.3;
            xs = xs(keep_idx);
            ys = ys(keep_idx);
            zs = zs(keep_idx);
            rc_values = RC(keep_idx);
            scatterers = double([xs, ys, zs]);
            %amplitudes = rc_values; 
            amplitudes = randn(size(rc_values));

        case {'empty_ellip', 'mesh', 'three_ellip', 'two_chambers'}
            % Load data
            load(scatterer_path, 'scatterers', 'rc_values');
            scatterers = double(scatterers)/1000;
            amplitudes = randn(size(rc_values)); 
        
        otherwise
            error('Unknown mode: %s', mode);
        
    end

    % Align Scatterers
    beam_center = [0, 0, 0.1];
    scatterers = double(scatterers);
    scatterer_center = mean(scatterers, 1);
    shift = beam_center - scatterer_center;
    positions = scatterers + shift;

    % Check scatters
    fprintf(['Min X: ', num2str(min(positions(:,1) * 1000)), ' mm\n'])
    fprintf(['Max X: ', num2str(max(positions(:,1) * 1000)), ' mm\n'])
    fprintf(['Min Y: ', num2str(min(positions(:,2) * 1000)), ' mm\n'])
    fprintf(['Max Y: ', num2str(max(positions(:,2) * 1000)), ' mm\n'])
    fprintf(['Min Z: ', num2str(min(positions(:,3) * 1000)), ' mm\n'])
    fprintf(['Max Z: ', num2str(max(positions(:,3) * 1000)), ' mm\n'])

    rc_values = rc_values / max(rc_values(:));
    g = 40;
    rc_scaled = 10.^((g/20) * (rc_values - 1));
    rc_values = rc_scaled .* hypot(rand(size(rc_values)), rand(size(rc_values))) / sqrt(pi/2);

    % Generate and save scatter plot
    figure1 = figure('Visible', 'off');
    scatter3(positions(:,1)*100, positions(:,2)*100, positions(:,3)*100, 2, 20*log10(rc_values), 'filled');
    xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
    axis equal tight; colormap hot; clim([-40 0]); colorbar;
    saveas(figure1, fullfile(out_fig_dir, [scatterer_name, '_scatter.png']));
    close(figure1);

    % Precalculate time
    % Time delay for impulse response alignmnet
    [~,I]=max(abs(hilbert(irLUT(:,1))));
    t0_shift=I/fs;

    % Min and max t limits
    tmin=(2*min(positions(:,3))-max(elementPos(:,3))-max(elementPos(:,3)))/c-t0_shift;
    dist_maxi=sqrt((max(sqrt(positions(:,1).^2+positions(:,2).^2))+max(sqrt(cat(1,elementPos(:,1),elementPos(:,1)).^2+cat(1,elementPos(:,2),elementPos(:,2)).^2))).^2+...
        (max(cat(1,elementPos(:,3),elementPos(:,3)))+max(positions(:,3))).^2);
    tmax=2*dist_maxi/c+numel(irLUT(:,1))/fs;
    time_vec=tmin:1/fs:tmax;
    time_length=numel(time_vec);

    % Param Structure definition
    param.fs=single(fs);
    param.c=single(c);
    param.tmin=single(tmin);
    param.mstl=single(time_length);

    % Sort scatterers by angles for memory optimization
    angle_elemt_Emt_scat=round(atand(sqrt(sum((elementPos(1,1:2)-positions(:,1:2)).^2,2))./positions(:,3)));
    [~,I]=sort(angle_elemt_Emt_scat);
    positions=positions(I,:);
    amplitudes=amplitudes(I);

    % Begin Simulation loop
    sta_data=cell(1,nb_element);
    for num_emit=1:nb_element
        %disp(strcat("Emit event ",num2str(num_emit)));
        param.emitter=single(num_emit);
        rf_out=STA3D(single(irLUT),single(positions),single(amplitudes),single(elementPos),param,single(time_vec));
        sta_data{num_emit}=rf_out;
    end
    % Fill missing rf
    sta_data=[sta_data{:}];
    sta_data=reshape(sta_data,[],nb_element,nb_element);
    [sta_data] = fillEmptyRX(sta_data);

    %% Create the RF-data-like with the STA one
    fprintf('Creating RF-dwi data from STA...');
    % [vs, dels] = getVirtualSourcesPosition(elementPos, 60, 15, [7 7], param.c);
    [vs, dels] = getVirtualSourcesPosition(elementPos, 60, 20, [9 9], param.c);
    multiWaveforms = calc_dwi_delMatrix(1, dels, param.fs);

    for k=1:length(vs.x)
        emit_mat=multiWaveforms(k).tot;
        raw=0;
        for i=1:nb_element
            raw=raw+conv2(sta_data(:,:,i), emit_mat(:,i));
        end
        if (k==1)
            dwi_data = zeros([size(raw) length(vs.x)]);
        end
        dwi_data(:,:,k) = raw;
    end
    fprintf('DWI Data generation done\n');

    %% Run the ultraspy beamforming on GPU...
    fprintf("DWI beamforming...\n");
    %-- Creation of the param structure
    param=struct();
    param.fs = fs;
    param.c = c;
    param.f0=f0;
    param.t0=(tmin-t0_shift);
    param.fnumber = [1.5 1.5];
    
    %-- beamforming grid definition
    x = linspace(-50,50,192);
    y = linspace(-50,50,192);
    z = linspace(50,150,192);
    % Get middle slices
    iy = round(length(y)/2); 
    iz = round(find(z >= 100, 1));

    transmit_positions=[vs.x vs.y -vs.z];
    [th, az] = meshgrid(linspace(-20, 20, 9), linspace(-20, 20, 9));
    th = th(:);
    az = az(:);
    ind = [ ];
    for i = -15:15:15
        for j = -15:15:15
            ind = [ind find(th==i & az==j)];
        end
    end

    % do the beamforming for all the data
    param.do_sommation_on_TX = 0;
    [dwi_beamf, ~, dwi_beamf_HR] = call_DAS_3D_IQ(dwi_data, elementPos, transmit_positions, x/1000, y/1000, z/1000, param, ind);


    % save 9 dw
    save(fullfile(out_file_dir, [scatterer_name, '_dwi.mat']), 'dwi_beamf');
    fprintf("DWI data saved from  %s\n", scatterer_name);


    figure2 = figure('Visible', 'off');
    subplot(211)
    imagesc(x, z, squeeze(abs(sum(dwi_beamf(:,:,:,iy))))); axis image
    title("Side view")
    subplot(212)
    imagesc(y, x, squeeze(abs(sum(dwi_beamf(:,iz,:,:))))); axis image
    title("Top view")
    saveas(figure2, fullfile(out_fig_dir, [scatterer_name, '_beamform.png']));
    close(figure2);
    fprintf("Image 2 saved from  %s\n", scatterer_name)

    % save target dws
    %fprintf('Size of dwi_beamf_HR %s\n', size(dwi_beamf_HR));
    save(fullfile(out_file_dir, [scatterer_name, '_tar.mat']), 'dwi_beamf_HR');
    fprintf("CF data saved\n");

    figure3 = figure('Visible', 'off');
    subplot(211)
    side = squeeze(abs(dwi_beamf_HR(:,:,iy)));
    side = max(side, -60);
    imagesc(x, z, side); axis image
    title("Side view")
    subplot(212)
    top = squeeze(abs(dwi_beamf_HR(iz,:,:)));
    top = max(top, -60);
    imagesc(y, x, top); axis image
    title("Top view")
    saveas(figure3, fullfile(out_fig_dir, [scatterer_name, '_tar.png']));
    close(figure3);
    fprintf("Image 3 saved from  %s\n", scatterer_name)

    elapsed_time = toc;
    hours = floor(elapsed_time / 3600);
    minutes = floor(mod(elapsed_time, 3600) / 60);
    seconds = mod(elapsed_time, 60);
    fprintf('PROCESS FINISHED. \n Total time DWI:  %02d:%02d:%05.2f for %d scatterers\n', hours, minutes, seconds, length(positions));

    % Full GPU reset
    reset(gpuDevice);

end