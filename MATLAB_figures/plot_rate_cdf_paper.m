clear; clc; close all;

repo_root = fileparts(mfilename("fullpath"));
results_path = fullfile(repo_root, "..", "results", "cdf_point.npz");
out_pdf = fullfile(repo_root, "cdf_paper.pdf");

d = npzload(results_path);
schemes = string(d.schemes);
rates = double(d.rates); % (S x nSeeds x rtLoops x K)

colors = lines(6);

figure; hold on; grid on;
plots = gobjects(numel(schemes), 1);

for i = 1:numel(schemes)
    samples = reshape(rates(i, :, :, :), 1, []);
    samples = samples(isfinite(samples));
    if isempty(samples)
        continue;
    end
    x_sorted = sort(samples);
    y = (1:numel(x_sorted)) / numel(x_sorted);

    [c, ls, ~, lw] = local_style_for_scheme(schemes(i), colors);
    plots(i) = plot(x_sorted, y, "Color", c, "LineStyle", ls, "LineWidth", max(lw, 1.8));
end

yl = ylim;
xl = [0 16];

plot([xl(1) xl(2)], [0.05 0.05], 'k--', 'LineWidth', 1);
plot([xl(1) xl(2)], [0.95 0.95], 'k--', 'LineWidth', 1);

% Add text labels
text(0.2*xl(2), 0.05, ' 5th percentile', 'VerticalAlignment', 'bottom', ...
    'HorizontalAlignment', 'right', 'FontSize', 20, 'FontName', 'Times');
text(0.95*xl(2), 0.95, ' 95th percentile', 'VerticalAlignment', 'top', ...
    'HorizontalAlignment', 'right', 'FontSize', 20, 'FontName', 'Times');

xlabel("Per-user Throughput (bits/s/Hz)");
ylabel("Empirical CDF");
legend(plots(isgraphics(plots)), local_legend_labels(schemes(isgraphics(plots))), ...
    "Location", "southeast", "Interpreter", "latex");
paper_style(gca);
set(gcf, "Position", [100 0 900 600]); % match legacy rate_convergence figure size

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
    % All curves use solid lines; the MA-DRL PA baseline is identified by
    % colour alone in the CDF plot (which is marker-less).
    switch string(scheme)
        case "greedy+robust"
            c = colors(1, :); ls = "-"; mk = "s"; lw = 1.25;
        case "greedy+oblivious"
            c = colors(2, :); ls = "-"; mk = "+"; lw = 1.25;
        case "naive+oblivious"
            c = colors(6, :); ls = "-"; mk = "d"; lw = 1.25;
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

