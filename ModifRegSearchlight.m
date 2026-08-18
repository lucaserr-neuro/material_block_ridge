%
% searchlight ridge encoding model
% Initialisation

    s = 1;                     
    n_runs = 8;
    n_searchlight_voxels = 50;
    lambda = 2;
    %measure = 'correlation'; %mae does not work for this anymore
    
    base_path = 'base/path';
    spm_file = sprintf('%s/sub-%02d/1stAnalysis/SPM.mat',base_path,s);
    
    % load data
    tool_mask = '/path/to/tool/ROI.nii';
    brain_mask =  'path/to/brain/mask';
    
    %I'm used to using Cosmo functions, I think they are nice
    ds      = cosmo_fmri_dataset(spm_file,'mask',brain_mask);
    ds = cosmo_remove_useless_data(ds);
    ds_tool = cosmo_fmri_dataset(spm_file,'mask',tool_mask);
    
   
    
   % condition masks
    %condition order: material=4, colour=2, shape=6, text=7, tool=8
    % each condition appears once per run -> 8 samples per mask
    
    material_con = logical(repmat([0 0 0 1 0 0 0 0 0],1,n_runs)');
    colour_con   = logical(repmat([0 1 0 0 0 0 0 0 0],1,n_runs)');
    shape_con    = logical(repmat([0 0 0 0 0 1 0 0 0],1,n_runs)');
    text_con     = logical(repmat([0 0 0 0 0 0 1 0 0],1,n_runs)');
    tool_con     = logical(repmat([0 0 0 0 0 0 0 1 0],1,n_runs)');
    
    data_target_ROI = ds_tool.samples(tool_con,:);
    
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
    
 
    
    % searchlight
    
    searchlight_result = NaN(1,ncenters);
    searchlight_run_correlations = NaN(ncenters,n_runs);
    
    %The storing of weights:
    % Then the weights are stored as averages for each searchlight, so I
    % have a n_voxels x 4 matrix, storing the weight of each feature for
    % each voxel. This is not actually meaningful, but I would look later
    % into creating models where features are dropped to identify their
    % unique contribution

    searchlight_feature_weight_mean = NaN(ncenters,4);
    
    tic;
    for isphere = 1:ncenters
    
        [score,run_corr,W] = predict_ROI( ...
            sphere_data{isphere},data_target_ROI,labels_group,lambda);
    
        searchlight_result(isphere) = score;
        searchlight_run_correlations(isphere,:) = run_corr;
        
        fi = sphere_feature_indices{isphere};

        %Weigth means over each sphere
        searchlight_feature_weight_mean(isphere,:) = [ ...
        mean(abs(W(fi.material,:)),'all'), ...
        mean(abs(W(fi.colour,:)),'all'), ...
        mean(abs(W(fi.shape,:)),'all'), ...
        mean(abs(W(fi.text,:)),'all')];
    end
    fprintf('\n searchlight: %.2f s\n',toc);
    
    fprintf('result min/mean/max: %.4f/%.4f/%.4f\n', ...
        min(searchlight_result),mean(searchlight_result,'omitnan'),max(searchlight_result));
    


function [mean_score,run_correlations,weights] = predict_ROI( ...
    data_source,data_target,labels_group,lambda)

n = size(data_source,1);
m_source = size(data_source,2);
m_target = size(data_target,2);
%Allowing different number of voxels between source and target

run_correlations = NaN(n,1);
weights = NaN(m_source,m_target);

groups = unique(labels_group);
%I removed the zscoring options and kept the optimised formula only

for ig = 1:length(groups)
    
    group = groups(ig);
    test_mask  = (labels_group == group);
    indices_test  = find(test_mask);
    indices_train = find(~test_mask);

    X_train = data_source(indices_train,:);
    X_test = data_source(indices_test,:);
    Y_train = data_target(indices_train,:);
    Y_test = data_target(indices_test,:);

    %centering
    mean_X_train = mean(X_train,1);
    X_train = X_train - repmat(mean_X_train,size(X_train,1),1);
    mean_Y_train = mean(Y_train,1);
    Y_train = Y_train - repmat(mean_Y_train,size(Y_train,1),1);
    X_test = X_test  - repmat(mean_X_train,size(X_test,1),1);

    % ridge weights, avoiding  matrix inversion
    weights = (X_train'*X_train + lambda*eye(m_source)) \ (X_train'*Y_train);

    %predicting + adding the mean 
    Y_prediction = X_test*weights + repmat(mean_Y_train,size(X_test,1),1);

    %Got rid of mae, only calculating correlation of spatial pattern
    for itest = 1:length(indices_test)
        idx = indices_test(itest);
        run_correlations(idx) = corr( ...
            Y_test(itest,:)',Y_prediction(itest,:)','Rows','complete');
    end
        
end

mean_score = mean(run_correlations,'omitnan');

end