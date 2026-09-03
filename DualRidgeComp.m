    s = 1;%:22                     % subject
    n_runs = 8;
    n_searchlight_voxels = 50;
    lambda = 2;
    npermutations = 10000;
    measure = 'correlation';
    base_dir = '/media/proaction/data1/LUCA/TextMat';
    addpath(genpath('/home/proaction/Documents/MATLAB/CoSMoMVPA-master'))
addpath(genpath('/home/proaction/Documents/MATLAB/libsvm-3.36'))

    base_path = '/media/proaction/data1/LUCA/TextMat/Univariate/Normal';
    spm_file = sprintf('%s/sub-%02d/1stAnalysis/SPM.mat',base_path,s);
    
    % load data
    tool_mask = '/media/proaction/data1/LUCA/TextMat/Regression/ROIs/Tool_No_C_10mm_resampled.nii';
 %   place_mask = '/media/maria/Storage-8TB/PhD/WalidBlock/Regression/ROIs/Place_No_CSF_resampled.nii';
    brain_mask =  sprintf('%s/forFilipa/fmriprep/sub-%02d/func/sub-%02d_task-01ep2dboldprftools_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz', base_dir,s,s);
    
    
    ds      = cosmo_fmri_dataset(spm_file,'mask',brain_mask);
    ds = cosmo_remove_useless_data(ds);
    ds_tool = cosmo_fmri_dataset(spm_file,'mask',tool_mask);
    
    fprintf('whole brain: %d x %d, tool ROI: %d x %d\n', ...
        size(ds.samples,1),size(ds.samples,2), ...
        size(ds_tool.samples,1),size(ds_tool.samples,2));
    
   % % condition masks
    %
    % SPM condition order: material=4, colour=2, shape=6, text=7, tool=8
    % each condition appears once per run -> 8 samples per mask
    
    material_con = logical(repmat([0 0 0 1 0 0 0 0 0],1,n_runs)');
    colour_con   = logical(repmat([0 1 0 0 0 0 0 0 0],1,n_runs)');
    shape_con    = logical(repmat([0 0 0 0 0 1 0 0 0],1,n_runs)');
    text_con     = logical(repmat([0 0 0 0 0 0 1 0 0],1,n_runs)');
    tool_con     = logical(repmat([0 0 0 0 0 0 0 1 0],1,n_runs)');
    
    data_target_ROI = ds_tool.samples(tool_con,:);
    assert(size(data_target_ROI,1) == n_runs,'expected one target row per run');
    
    labels_group = 1:n_runs;
    
    % searchlight neighborhood
    nbrhood = cosmo_spherical_neighborhood(ds,'count',n_searchlight_voxels);
    ncenters = numel(nbrhood.neighbors);
    
    
    % precomputing source data (material|colour|shape|text) for every sphere
    sphere_data = cell(ncenters,1);
    sphere_feature_indices = cell(ncenters,1);
    
    tic;
    for isphere = 1:ncenters

        voxels = nbrhood.neighbors{isphere};
        sphere_samples = ds.samples(:,voxels);
    
        material_samples = sphere_samples(material_con,:);
        colour_samples   = sphere_samples(colour_con,:);
        shape_samples    = sphere_samples(shape_con,:);
        text_samples     = sphere_samples(text_con,:);
    
        sphere_data{isphere} = single([material_samples,colour_samples,shape_samples,text_samples]);
    
        nvox = size(material_samples,2);
        sphere_feature_indices{isphere} = struct( ...
            'material',1:nvox,'colour',nvox+(1:nvox), ...
            'shape',2*nvox+(1:nvox),'text',3*nvox+(1:nvox));
    end
    fprintf('\nprecompute: %.2f s\n',toc);
    
    assert(size(sphere_data{1},1) == size(data_target_ROI,1), ...
        'source/target run count mismatch');
    

searchlight_result = NaN(1,ncenters);
searchlight_run_correlations = NaN(ncenters,n_runs);
p_values = NaN(1,ncenters);

% Start parallel pool
if isempty(gcp('nocreate'))
    parpool(94);   % adjust to your RAM/CPU
end

parfor isphere = 1:ncenters

    % Source data
    data_source = sphere_data{isphere};

    % Precompute - function that creates the woodbury matrix per sphere
    dprecomp = precompute_sphere_ridge_dual( ...
        data_source, data_target_ROI, lambda);
    
    %when perm = 1 then -> no run shuffling, 
    %estimation
    perm = 1:n_runs;
    
    [observed_score, run_correlations] = ...
        predict_ROI_dual(dprecomp, perm, measure);

    n_exceed = 0;

    % Reproducible stream for this sphere
    stream = RandStream('Threefry', ...
        'Seed', 12345 + isphere);

    for iperm = 1:npermutations

        perm = randperm(stream,n_runs);

        perm_score = predict_ROI_dual( ...
            dprecomp, perm, measure);

        if perm_score >= observed_score
            n_exceed = n_exceed + 1;
        end
    end
   
    % Store results
    searchlight_result(isphere) = observed_score;
    searchlight_run_correlations(isphere,:) = ...
        run_correlations';

    p_values(isphere) = ...
        (n_exceed + 1)/(npermutations + 1);

end


    save('results/p_values_10000perms_sub1.mat',"p_values")

%below to get the right dimensions to give to cosmo dataset and saving
%p_logged map
ds_b = cosmo_fmri_dataset(sprintf('%s/sub-%02d/1stAnalysis/beta_0001.nii',base_path,s),'mask', brain_mask);

P_val_ds = ds_b;
P_val_ds = cosmo_remove_useless_data(P_val_ds);
P_val_ds.samples = p_values;
P_val_ds.samples = -10*log(P_val_ds.samples); % for visualisation purposes
cosmo_map2fmri(P_val_ds,'p_vals_sub-01.nii')