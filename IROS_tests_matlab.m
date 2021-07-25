% Trust regions coordinate descent solver to minimize the energy in Eq. (3)
% from the paper Multiview Euclidean Distance Matrix Matching

% Dependencies: manopt

% NOTE: hotfix for manopt needed. In manopt\manifolds\multinomial\doubly_stochastic.m,
% copy line 72 (gap = max(abs(row .* d_1 - 1));) and paste it under 86 (if ...)

% NOTE: CURRENTLY USING FINITE DIFFERENCE APPROXIMATED TRUST REGIONS
warning('off', 'manopt:getHessian:approx')
clear all; close all; clc;
%%%%%%%%%%%%%%%%%%%%%% PROBLEM FORMULATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rng(15);

% number of observations
k = 2;
% dimensionality of underlying data
d = 3;
% number of points in target point cloud
m = 500;
% vector where n(c) represents num observed points for observation c
n = zeros(1, k);

% observed distance matrices
D_ = cell(1, k);
S_ = cell(1, k);

%%%%%%%%%%%%%%%% Hyperparameters

% promotes the norm of the P_i matrices
% lambda = 0;
% promotes the 6D points to be in 3D hyperplane
lambda2 = 1;

% Loads point cloud data for Stanford Bunny
bunny = load('./data/bunnyData.mat');
bunnyData = bunny.bunnyData;
% cell used to store coordinates as doubles instead of floats
Xpart = cell(1, k);
% difference operator to extract coplanality from rank condition
% transform matrix with rows (row1, row2, row3, ...) to (row2 - row1. row3 - row1, ...)
Diff_ = cell(1, k);
% Q_{c} is orthogonal part of QR factorization of Diff_{c} * Xpart{c}
Q_ = cell(1, k);

P_view = eye(m);
P_view  = P_view(randperm(m),:);

for c = 1:k
    % Store linearization of upper triangular part of observed EDM
    n(c) = size(bunnyData.Xpart{1}, 1);

    % construction of S_i matrices, don't want this explicitly in final algorithm
    S_{c} = eye(n(c));
    S_{c}((n(c) + 1):m, :) = zeros(m - n(c), n(c));

    Xpart{c} = zeros(size(bunnyData.Xpart{1}));
    Xpart{c}(:, :) = bunnyData.Xpart{c}(:, :);
    if c == 2
        Xpart{c} = P_view' * Xpart{c};
    end

    Diff_{c} = eye(n(c)) - (1/n(c))*ones(n(c), n(c));

    [Q_{c}, R] = qr_unique(Diff_{c} * Xpart{c});
    D_{c} = symToDist(Xpart{c});
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Current testing model: we only consider the 2-view case. To simplify matters
% even further, we take two overlapping samples of one view of the Stanford bunny,
% explicitly. This way, we know the explicit permutation that matches points between
% these two views, and we can easily test for arbitrary permutations how many points
% it matches correctly (and how close some assignments get).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% manifolds to optimize over permutation matrices
man_P = multinomialdoublystochasticfactory(m);
problem_P.M = man_P;
problem_P.cost = @(X) costFunc_P(X, Xpart{1}, D_{2}, Diff_{1}, Q_{2}, lambda2);
problem_P.egrad = @(X) egrad_P(X, Xpart{1}, D_{2}, Diff_{1}, Q_{2}, lambda2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TESTING WITH KNOWN SYNCHRO %%%%%%%%%%%%%%%%%%%%%%%
% current_iterate_P{1} = eye(m);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% problem structure for EDM optimization
% problem_D.cost = @(X) costFunc_D(X, current_iterate_P, D_, S_, Xpart{1}, 2);
% problem_D.grad = @(X) man_D.proj(X, egrad_D(X, current_iterate_P, D_, S_, Xpart{1}, 2));

% General options for trust regions
% options.Delta0 = 10;
% options.Delta_bar = 10 * 2^8;
options.maxiter = 200;
% options.maxinner = 200;
% options.maxtime = inf;
% options.tolgradnorm = 5e-6;

% tracks the current trust region radius for each optimization
Delta_D = 0.1;
Delta_P = cell(1, k);

for c = 1:k
    Delta_P{c} = 10;
end

checkgradient(problem_P)
[P_opt, cost_P, stats_P] = trustregions(problem_P, (1/m)*ones(m,m), options);

for r = 1:m
    [argval, k] = max(P_opt(:,r));
    P_opt(:,r) = zeros(m,1);
    P_opt(k,r) = 1;
end

sum(sum(abs(P_opt - P_view)))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% METHODS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Converts tall matrix of size (m x d) to EDM using Eq (1) of main paper
function dist = symToDist(Y)
    % get size of target EDM
    m = size(Y, 1);

    % reconstructed PSD matrix
    G = Y * Y';
    diagonal = diag(G); % shape (m x 1)

    % corresponds to 2nd term in Eq (1) of main paper
    diagRows = zeros(m, m);
    % corresponds to 1st term in Eq (1) of main paper
    diagCols = zeros(m, m);

    for k = 1:m
        diagRows(k, :) = diagonal';
    end

    for k = 1:m
        diagCols(:, k) = diagonal;
    end

    dist = diagRows + diagCols - 2 * G;
end



% Same as costFunc_D, but specify a single P{index} to be treated as a variable,
% inputted as P_var
function c = costFunc_P(P_i, D_Y, D_i, Diff, Q_i, lambda2)
    % k = length(P);
    % c = 0;
    D = symToDist(D_Y);
    n_i = size(D_i, 1);

    c = norm(P_i' * D * P_i - D_i, 'fro')^2 ...
        + lambda2 * norm(Diff * P_i' * D_Y - (Q_i * Q_i') * Diff * P_i' * D_Y, 'fro')^2;

    % TEMP COST that only calculates for the 2nd permutation
    % a = 2;
    % c = norm(S_{a}' * P_var' * D * P_var * S_{a} - D_{a}, 'fro')^2 - lambda * norm(P_var, 'fro')^2;

end


% Computes gradient of costFunc_P w.r.t. P_i
function g = egrad_P(P_i, D_Y, D_i, Diff, Q_i, lambda2)
    D = symToDist(D_Y);
    n_i = size(D_i, 1);

    % g = 4 * D * P_i * S_i * (S_i' * P_i' * D * P_i * S_i - D_i) * S_i' - 2 * lambda * P_i;
    % + 2 * lambda2 * (D_Y * D_Y') * P_i * S_i * Diff' * (eye(n_i - 1) - Q_i * Q_i') * Diff * S_i';

    %%%%%%%%%%%%%%%% NEW ALT COST FUNC GRAD

    % g = -2 * D * P_i * S_i * D_i * S_i' - 2 * lambda * P_i * (S_i * S_i') + 2 * lambda2 * (D_Y * D_Y') * P_i * S_i * (eye(n_i) - Q_i * Q_i') * S_i';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    g = 4 * D * P_i * (P_i' * D * P_i - D_i) ...
        + 2 * lambda2 * (D_Y * D_Y') * P_i * Diff' * (eye(n_i) - Q_i * Q_i') * Diff;
end

function h = ehess_P(V, D_, lambda)
    h = 2 * (D_{1} * D_{1} * V + V * D_{2} * D_{2} - 2 * D_{1} * V * D_{2}) - 2 * lambda * V; ...
end