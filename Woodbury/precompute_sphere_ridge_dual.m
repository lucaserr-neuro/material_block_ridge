function dprecomp = precompute_sphere_ridge_dual(data_source, data_target, lambda)
% precompute_sphere_ridge_dual  Dual/kernel-ridge form of precompute_sphere_ridge.
%
% Uses the exact identity (X'X + lambda*I_m)^-1 * X' = X' * (XX' + lambda*I_n)^-1
% so that ALL per-permutation linear algebra operates on n x n (here 8x8 or
% smaller) matrices instead of m_source x m_source (here 200x200) matrices.
% This is mathematically exact -- not an approximation -- and should match
% predict_ROI to floating-point precision.
%
% The only step that ever touches the full m_source-dimensional data is the
% single Gram matrix computation below (data_source*data_source'), which is
% O(n^2 * m_source) and done ONCE per sphere. Every permutation after that
% only manipulates n x n / (n-1) x (n-1) matrices.
%
% data_source: n x m_source (original, unpermuted order)
% data_target: n x m_target (never permuted)

[n, ~] = size(data_source);
m_target = size(data_target,2);

Gxx_full = data_source * data_source';   % n x n Gram matrix -- the ONLY place
                                          % the full voxel data gets touched
RowSum   = sum(Gxx_full,2);              % n x 1
TotalSum = sum(Gxx_full(:));             % scalar

Sy_full = sum(data_target,1);
mu_y = zeros(n, m_target);
for ig = 1:n
    mu_y(ig,:) = (Sy_full - data_target(ig,:)) / (n-1); %computes mean to later demean
end

R = cell(n,1);          
idx_store = cell(n,1);  %  original rows are training when j is excluded
k_test_store = cell(n,1);

for j = 1:n

    %How similar is run j to all runs (including itself)?
    RowDot_j = (RowSum - Gxx_full(:,j)) / (n-1);          % n x 1, x_a . mu_x_j

    %How similar is the group average to itself?(???)
    c_j = (TotalSum - 2*RowSum(j) + Gxx_full(j,j)) / (n-1)^2;  % ||mu_x_j||^2

    % centered Gram matrix under mu_x_j (same trick as kernel-PCA centering)
    % Centering the values, by subtracting the group average
    Ktilde_j = Gxx_full - RowDot_j*ones(1,n) - ones(n,1)*RowDot_j' + c_j;

    idx = setdiff(1:n, j);
    K_train_j = Ktilde_j(idx, idx);     % (n-1) x (n-1)
    k_test_j  = Ktilde_j(idx, j);       % (n-1) x 1

    R{j} = chol(K_train_j + lambda*eye(n-1)); %more efficient factoring
    idx_store{j} = idx;
    k_test_store{j} = k_test_j;
end

dprecomp.n = n;
dprecomp.m_target = m_target;
dprecomp.data_target = data_target;
dprecomp.mu_y = mu_y;
dprecomp.R = R;
dprecomp.idx_store = idx_store;
dprecomp.k_test_store = k_test_store;
dprecomp.lambda = lambda;

end