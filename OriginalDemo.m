%
% demo for prediction of ROI B from ROI A
%

rand('state',1143);
colormap('parula');

% design parameters
nconditions = 8;
nminiblocks = 8;
n = nconditions * nminiblocks;

% # of voxels behaving the same every time each condition appears
m_common = 10;
% # of voxels doing things unrelated to each condition
m_different_A = 20;
m_different_B = 20;

% total # of voxels per ROI
m_A = m_common + m_different_A;
m_B = m_common + m_different_B;

% standard deviation for gaussian noise
noise_stdv = 0.1;

% measure of success for each voxel
measure = 'correlation';
%measure = 'MAE';

%
% generate data from two ROIs
%
% activation in each ROI is generated as a combination of
% 1) a template for consistent voxel behavior across conditions (repeats across miniblocks)
% 2) voxels that may do different things for the same condition across miniblocks
% + gaussian noise on top
% 
% the activation is generate U(0,1), which might make patterns a bit "busy".
% you can change it to be something else)

%% ROI A - assemble dataset, with voxels behaving consistently coming first

data_ROI_A = zeros(n,m_A);

% template for consistent voxel behaviour across 8 conditions in ROI A
template_common_ROI_A = rand(nconditions,m_common);

idx = 1;
for imb = 1:nminiblocks
    indices = idx:(idx+nconditions-1);
    data_ROI_A(indices,:) = [template_common_ROI_A,rand(nconditions,m_different_A)];
    idx = idx + nconditions;
end

% final data has gaussian noise on top
data_ROI_A = data_ROI_A + randn(size(data_ROI_A))*noise_stdv;


%% ROI B - assemble dataset, with voxels behaving consistently coming first

data_ROI_B = zeros(n,m_B);

% template for consistent voxel behaviour across 8 conditions in ROI B
template_common_ROI_B = rand(nconditions,m_common);

idx = 1;
for imb = 1:nminiblocks
    indices = idx:(idx+nconditions-1);
    data_ROI_B(indices,:) = [template_common_ROI_B,rand(nconditions,m_different_B)];
    idx = idx + nconditions;
end

% final data has gaussian noise on top
data_ROI_B = data_ROI_B + randn(size(data_ROI_B))*noise_stdv;

%% everything else is ROI independent

% condition labels - 1:8 twice, once for each miniblock
labels_condition = repmat((1:nconditions)',nminiblocks,1);

% for cross-validation - 1 for the first miniblock, 2 for the second, etc
tmp = repmat(1:nminiblocks,nconditions,1);
labels_group = tmp(:);

if 1
    clf;
    subplot(3,1,1);
    imagesc(data_ROI_A);
    title('data ROI A')
    subplot(3,1,2);
    imagesc(data_ROI_B);
    title('data ROI B')
    subplot(3,1,3);
    imagesc(corr(data_ROI_A,data_ROI_B),[-1,1]);
    title('correlation between voxels in A and voxels in B');
    axis square; colorbar('vert');
    fprintf('press any key to continue\n'); pause
end

%
% run the prediction
%
% there are two implementations
% 1) efficient matrix algebra - predict all the voxels in one go
% 2) MATLAB ridge - loop over voxels to predict each one separately
% 
% they are mostly equivalent if 1) is set to z-score each voxel in the training set,
% but right now it just centers it, since that is all that is needed and more stable
%

tic; fprintf('run the crossvalidation loop using matrix algebra - ');
[measure_per_voxel,predictions,weights] = predict_ROI( data_ROI_A,data_ROI_B,labels_condition,labels_group,measure,1);
fprintf(' %f seconds\n\n',toc);

tic; fprintf('run the crossvalidation loop using MATLAB ridge - ');
[measure_per_voxel_slow,predictions_slow] = predict_ROI( data_ROI_A,data_ROI_B,labels_condition,labels_group,measure,0);
fprintf(' %f seconds\n\n',toc);

%% compare results for each voxel

fprintf('measure: %s\n',measure);
fprintf('voxel\tfast\tridge\n');
for v = 1:m_B
    fprintf('%d\t%1.4f\t%1.4f\n',v,measure_per_voxel(v),measure_per_voxel_slow(v));
end

if 1
    vmin = min(data_ROI_B(:));
    vmax = max(data_ROI_B(:));
    
    clf;
    subplot(3,1,1);
    imagesc(data_ROI_B,[vmin,vmax]); colorbar('vert');
    title('actual ROI B data');
    subplot(3,1,2);
    imagesc(predictions,[vmin,vmax]); colorbar('vert');
    title('predicted ROI B data (fast)');
    subplot(3,1,3);
    imagesc(predictions_slow,[vmin,vmax]); colorbar('vert');
    title('predicted ROI B data (MATLAB ridge)');
    fprintf('press any key to continue\n'); pause
end

%
% generate a p-value for the result in each voxel
%
%
%            (#results >= true) + 1
% p-value = ------------------------
%             #permutations + 1
%

fprintf('running permutation test ');

tic;

npermutations = 10000;

permutation_results = zeros(npermutations,m_B);

for ip = 1:npermutations
    
    if rem(ip,100) == 0
        fprintf('%d ',ip);
    end
    
    %% permutation here is shuffling the source ROI, so there is no relation
    %% between activation in source and target

    shuffled_data_ROI_A = zeros(size(data_ROI_A));
    
    for ib = 1:nminiblocks
        indices = find(labels_group == ib);
            
        % since there are #conditions examples in each group
        permutation = randperm(nconditions);
        
        shuffled_data_ROI_A(indices,:) = data_ROI_A(indices(permutation),:);        
    end
    
    %% run the entire analysis on shuffled data
    
    permutation_results(ip,:) = predict_ROI( shuffled_data_ROI_A,data_ROI_B,labels_condition,labels_group,measure,1);    
end

time_stop = toc;

fprintf('\n%d permutations in %f seconds\n',npermutations,toc);

%% compute p-values

% binary mask for whether the result in each voxel/permutation is better than true result
switch measure 
  case {'correlation'}
    tmp = permutation_results >= repmat(measure_per_voxel,npermutations,1);
  case {'MAE'}
    tmp = permutation_results <= repmat(measure_per_voxel,npermutations,1);
  otherwise
    assert false;
end
    
pvalues = (sum(tmp,1) + 1) / (npermutations+1);

fprintf('\npermutation p-values of voxels in ROI B\n');
for v = 1:m_B
    fprintf('%d\t%1.4f\n',v,pvalues(v));
end

%
% self-contained code for cross-validation loop
% (to allow re-use in permutation tests)
% - measure is either 'correlation' or 'MAE' between each voxel prediction and ground truth
% - optimized == 1 is a 
%

function [measure_per_voxel,predictions, weights] = predict_ROI( data_source_ROI,data_target_ROI,labels,labels_group,measure,optimized,permute)

    % derive a few things
    
    % should have same number of examples
    n_source = size(data_source_ROI,1);
    n_target = size(data_target_ROI,1);
    if n_source ~= n_target
        assert(false);
    else
        n = n_source;
    end
    
    % allow different numbers of voxels, when you re-use this code
    m_source = size(data_source_ROI,2);
    m_target = size(data_target_ROI,2);
    
    unique_labels = unique(labels);
    nconditions = length(unique_labels);
    
    groups = unique(labels_group);
    ngroups = length(groups);
    
    %% some binary arguments change how the code works overall 
    %% - optimized (efficient code using matrix algebra, predicts all voxels at once)
            
    if optimized
        % my implementation requires one of these normalizations
        % the maths only require centering, but zscoring is needed
        % to get results close to those with MATLAB ridge
        % (zscore is a bit brittle if there are too few examples)
        normalization = 'center';
        %normalization = 'zscore';
    else
        % uses MATLAB ridge to predict each voxel
    end
    
  
    %% cross-validation loop

    predictions = zeros(n,m_target);
    
    for ig = 1:ngroups

        % find indices for train/test examples in this fold
        group = groups(ig);
        mask = (labels_group == group);        
        indices_test  = find( mask); ntest  = length(indices_test);
        indices_train = find(~mask); ntrain = length(indices_train);
  
        % train the ridge regression
        % - direct prediction of all the voxels in target_ROI from source_ROI
        % - lambda fixed for now - can use a cross-validation call within the training set to decide
        % - using the matrix algebra so you can see what's happening

        lambda = 1;
        
        X_train = data_source_ROI(indices_train,:);
        X_test  = data_source_ROI(indices_test ,:);
        Y_train = data_target_ROI(indices_train,:);
        Y_test  = data_target_ROI(indices_test ,:);

        % use either an optimized call to predict all voxels at once
        % or one MATLAB ridge call to predict each voxel
        if optimized
            % center them - models will not need a column of ones
            mean_X_train = mean(X_train,1);
            X_train = X_train - repmat(mean_X_train,size(X_train,1),1);
            mean_Y_train = mean(Y_train,1);
            Y_train = Y_train - repmat(mean_Y_train,size(Y_train,1),1);
        
            % center test set using the mean from the training set
            X_test = X_test - repmat(mean_X_train,size(X_test,1),1);
            
            if normalization == 'zscore'
                % also divide each voxel by its standard deviation
                % (a bit brittle if there are too few examples)
                stdv_X_train = std(X_train,0,1);
                X_train = X_train ./ repmat(stdv_X_train,size(X_train,1),1);
                stdv_Y_train = std(Y_train,0,1);
                Y_train = Y_train ./ repmat(stdv_Y_train,size(Y_train,1),1);                

                % divide test set using the stdv from the training set
                X_test = X_test ./ repmat(stdv_X_train,size(X_test,1),1);
            end
                
        
            % - by the book, OLS weights are inv(X_train'*X_train) * X_train' * Y_train
            % - ridge weights are            inv(X_train'*X_train + lambda*eye(m)) * X_train' * Y_train
            % - avoiding matrix inversion    (X_train'*X_train + lambda*eye(m_source)) \ (X_train'*Y_train)
        
            % get the weights to predict all voxels, in one go
            weights = (X_train'*X_train + lambda*eye(m_source)) \ (X_train'*Y_train); 
        
            % predict from centered test set, 
            predictions(indices_test,:) = X_test * weights; 
        
            % add the mean back to the prediction
            predictions(indices_test,:) = predictions(indices_test,:) + repmat(mean_Y_train,size(Y_test,1),1);

            if normalization == 'zscore'
                % also rescale based on the standard deviation
                predictions(indices_test,:) = predictions(indices_test,:) .* repmat(stdv_Y_train,size(Y_test,1),1);
            end
            
        else
            % use MATLAB ridge, one voxel at a time
            % (this doesn't do any normalization of the voxels,
            % it's the B0 in "help ridge")
            
            for v = 1:size(Y_train,2)                
                v_weights = ridge(Y_train(:,v),X_train,lambda,0);
                
                predictions(indices_test,v) = X_test * v_weights(2:end) + v_weights(1);
            end
        end
            
        
    end 
    
    %% compute measure
    
    switch measure
      case {'MAE'}
        % compute MSE for every voxel at once
        measure_per_voxel =  sum(abs(data_target_ROI - predictions),1) ;
      case {'correlation'}
        % to compute for every voxel at once
        % 1) z-score voxels in each matrix
        tmp_data_target = zscore_matrix(data_target_ROI);
        tmp_predictions = zscore_matrix(predictions);
        % 2) compute correlation by elementwise multiplication of each voxel
        measure_per_voxel = sum(tmp_data_target.*tmp_predictions,1)/(n-1);
      otherwise
        print('error: unknown measure %s',measure); assert(false);
    end
end

%% ancillary function for scoring matrix

function [matrix_zscored] = zscore_matrix(matrix)
    
    [n,m] = size(matrix);
    
    matrix_mean = mean(matrix,1);
    matrix_stdv = std(matrix,0,1);
    
    matrix_zscored = matrix - repmat(matrix_mean,n,1);
    matrix_zscored = matrix_zscored ./ repmat(matrix_stdv,n,1); 
    
end