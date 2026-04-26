clear; clc; close all;

repo_root = fileparts(mfilename("fullpath"));
results_path = fullfile(repo_root, "..", "results", "K_sweep.npz");
out_pdf = fullfile(repo_root, "K_sweep_paper.pdf");

d = npzload(results_path);
schemes = string(d.schemes);
K = double(d.Ks(:)).';
thr = double(d.throughput); % (S x nK x nSeeds)
mean_thr = mean(thr, 3);

% Match legacy style (Times + lines palette + markers).
colors = lines(6);

figure; hold on; grid on;
plots = gobjects(numel(schemes), 1);

for i = 1:numel(schemes)
    [c, ls, mk, lw] = local_style_for_scheme(schemes(i), colors);
    plots(i) = plot(K, mean_thr(i, :), ...
        "Color", c, "LineWidth", lw, "LineStyle", ls, ...
        "Marker", mk, "MarkerSize", 8, "MarkerFaceColor", "none", "MarkerEdgeColor", c);
end

xlabel("Number of Users $K$", "Interpreter", "latex");
ylabel("Aggregate Throughput (bits/s/Hz)");
ylim([20 160])
legend(plots, local_legend_labels(schemes), "Location", "northwest", "Interpreter", "latex");
paper_style(gca);

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
    % Keep this close to legacy "lines()" look, with proposed highlighted.
    switch string(scheme)
        case "greedy+robust"      % proposed
            c = colors(1, :); ls = "-"; mk = "s"; lw = 1.25;
        case "greedy+oblivious"   % closest competitor
            c = colors(2, :); ls = "-"; mk = "+"; lw = 1.25;
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

