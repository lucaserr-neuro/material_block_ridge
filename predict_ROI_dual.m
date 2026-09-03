function [mean_score, run_correlations] = predict_ROI_dual(dprecomp, perm, measure)
% predict_ROI_dual  See precompute_sphere_ridge_dual for the derivation.

if ~strcmpi(measure,'correlation')
    error('only correlation is implemented in the dual path');
end

n = dprecomp.n;
data_target = dprecomp.data_target;
mu_y = dprecomp.mu_y;

% inv_perm(j) = ig such that perm(ig) = j
[~, inv_perm] = sort(perm);

run_correlations = NaN(n,1);

for ig = 1:n
    j = perm(ig);

    idx = dprecomp.idx_store{j};             % original rows in the training set for this j
    positions_for_idx = inv_perm(idx);       % which fold position each training row occupies
                                              % under THIS permutation
    Y_train_ordered = data_target(positions_for_idx,:);   % (n-1) x m_target, order-matched to idx

    R = dprecomp.R{j};
    alpha = R \ (R' \ Y_train_ordered);      % (n-1) x m_target

    prediction = dprecomp.k_test_store{j}' * alpha + mu_y(ig,:);   % 1 x m_target

    run_correlations(ig) = corr(data_target(ig,:)', prediction', 'Rows','complete');
end

mean_score = mean(run_correlations,'omitnan');

end