function annotate_max_improvement(ax, x, y_ours, y_others, varargin)
%ANNOTATE_MAX_IMPROVEMENT Add arrow/text for max % gain vs best competitor.
%   y_others is (nOtherCurves x nX) or (nX x nOtherCurves).
%
%   Optional name-value:
%     "TextFormat" (default "%.1f%%")
%     "FontSize"   (default 20)
%     "XOffset"    (default 0.0) Horizontal shift of the arrow & textbox in
%                  normalized figure units. Useful when the max occurs at
%                  the edge of the x-axis and the textbox would otherwise
%                  spill outside the plot area.

    p = inputParser;
    p.addParameter("TextFormat", "%.1f%%", @(s) ischar(s) || isstring(s));
    p.addParameter("FontSize", 20, @(v) isnumeric(v) && isscalar(v));
    p.addParameter("XOffset", 0.0, @(v) isnumeric(v) && isscalar(v));
    p.parse(varargin{:});

    if nargin < 1 || isempty(ax)
        ax = gca;
    end

    x = x(:).';
    y_ours = y_ours(:).';

    Y = y_others;
    if size(Y, 2) ~= numel(x) && size(Y, 1) == numel(x)
        Y = Y.';
    end
    if size(Y, 2) ~= numel(x)
        return;
    end

    best_other = max(Y, [], 1);
    denom = max(best_other, eps);
    imp_pct = 100 * (y_ours - best_other) ./ denom;
    [imp_best, idx] = max(imp_pct);
    if ~isfinite(imp_best) || imp_best <= 0
        return;
    end

    x0 = x(idx);
    y0 = best_other(idx);
    y1 = y_ours(idx);

    [xn, y0n] = local_data_to_norm(ax, x0, y0);
    [~,  y1n] = local_data_to_norm(ax, x0, y1);
    if any(~isfinite([xn, y0n, y1n]))
        return;
    end

    xn = xn + p.Results.XOffset;

    ylo = min(y0n, y1n);
    yhi = max(y0n, y1n);

    annotation("doublearrow", [xn xn], [ylo yhi]);
    ymid = 0.5 * (ylo + yhi);
    txt_h = 0.06;
    txt_y = min(max(ymid - 0.5 * txt_h, 0.02), 0.98 - txt_h);
    txt_x = min(max(xn + 0.01, 0.02), 0.98 - 0.12);
    annotation("textbox", [txt_x, txt_y, 0.12, txt_h], ...
        "String", sprintf(p.Results.TextFormat, imp_best), ...
        "EdgeColor", "none", ...
        "FontSize", p.Results.FontSize, ...
        "FontName", "Times");
end

function [xn, yn] = local_data_to_norm(ax, xd, yd)
    fig = ancestor(ax, "figure");
    oldUnitsAx = ax.Units;
    oldUnitsFig = fig.Units;
    ax.Units = "normalized";
    fig.Units = "normalized";
    pos = ax.Position;
    ax.Units = oldUnitsAx;
    fig.Units = oldUnitsFig;

    xl = xlim(ax); yl = ylim(ax);
    xn = pos(1) + pos(3) * ( (xd - xl(1)) / (xl(2) - xl(1)) );
    yn = pos(2) + pos(4) * ( (yd - yl(1)) / (yl(2) - yl(1)) );
end

