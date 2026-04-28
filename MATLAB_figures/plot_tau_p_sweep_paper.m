clear; clc; close all;

repo_root = fileparts(mfilename("fullpath"));
results_path = fullfile(repo_root, "..", "results", "tau_p_sweep.npz");
out_pdf = fullfile(repo_root, "tau_p_sweep_paper.pdf");

d = npzload(results_path);
schemes = string(d.schemes);
tau_p = double(d.tau_ps(:)).';
thr = double(d.throughput); % (S x nTau x nSeeds)
mean_thr = mean(thr, 3);

colors = lines(6);

figure; hold on; grid on;
plots = gobjects(numel(schemes), 1);
for i = 1:numel(schemes)
    [c, ls, mk, lw] = local_style_for_scheme(schemes(i), colors);
    plots(i) = plot(tau_p, mean_thr(i, :), ...
        "Color", c, "LineWidth", lw, "LineStyle", ls, ...
        "Marker", mk, "MarkerSize", 8, "MarkerFaceColor", "none", "MarkerEdgeColor", c);
end

xlabel("Number of Pilots $\tau_\mathrm{p}$", "Interpreter", "latex");
ylabel("Aggregate Throughput (bits/s/Hz)");
legend(plots, local_legend_labels(schemes), "Location", "northwest", "Interpreter", "latex");
paper_style(gca);
set(gcf, "Position", [100 0 900 600]); % match legacy rate_convergence figure size

idx_ours  = find(schemes == "greedy+robust", 1);
idx_naive = find(schemes == "naive+oblivious", 1);
if ~isempty(idx_ours)
    others = mean_thr; others(idx_ours, :) = -inf;
    annotate_max_improvement(gca, tau_p, mean_thr(idx_ours, :), others, "TextFormat", "%.1f%%");
end
% Also draw the max % gain of the proposed scheme over the MA-DRL PA
% baseline (naive+oblivious) at its own peak tau_p, similar to the
% existing arrow against the closest competitor.
if ~isempty(idx_ours) && ~isempty(idx_naive)
    annotate_max_improvement(gca, tau_p, mean_thr(idx_ours, :), ...
        mean_thr(idx_naive, :), "TextFormat", "%.1f%%");
end

exportgraphics(gcf, out_pdf, "ContentType", "vector", "BackgroundColor", "none");

function labels = local_legend_labels(schemes)
    labels = strings(size(schemes));
    for i = 1:numel(schemes)
        labels(i) = local_label_for_scheme(schemes(i));
    end
end

function s = local_label_for_scheme(scheme)
    switch string(scheme)
        case "greedy+robust"
            s = "Proposed Algorithm";
        case "greedy+oblivious"
            s = "CF-WMMSE, Proposed PA";
        case "naive+oblivious"
            s = "CF-WMMSE, MA-DRL PA";
        case "random+oblivious"
            s = "CF-WMMSE, Random PA";
        case "greedy+rzf"
            s = "RZF, Proposed PA";
        case "greedy+mrt"
            s = "MRT, Proposed PA";
        otherwise
            s = scheme;
    end
end

function [c, ls, mk, lw] = local_style_for_scheme(scheme, colors)
    % `naive+oblivious` is the simplified DRL baseline (Oh et al.); we draw
    % it with a dashed line and a diamond marker to distinguish it from the
    % proposed-PA family which uses solid lines.
    switch string(scheme)
        case "greedy+robust"
            c = colors(1, :); ls = "-"; mk = "s"; lw = 1.25;
        case "greedy+oblivious"
            c = colors(2, :); ls = "-"; mk = "+"; lw = 1.25;
        case "naive+oblivious"
            c = colors(6, :); ls = "--"; mk = "d"; lw = 1.25;
        case "random+oblivious"
            c = colors(4, :); ls = "-"; mk = "o"; lw = 1.25;
        case "greedy+rzf"
            c = colors(3, :); ls = "-"; mk = "*"; lw = 1.25;
        case "greedy+mrt"
            c = colors(5, :); ls = "-"; mk = "^"; lw = 1.25;
        otherwise
            c = colors(6, :); ls = "-"; mk = "o"; lw = 1.25;
    end
end

