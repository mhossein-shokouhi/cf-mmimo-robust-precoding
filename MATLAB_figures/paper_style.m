function paper_style(ax)
%PAPER_STYLE Apply a "paper-ready" style matching legacy plots.

    if nargin < 1 || isempty(ax)
        ax = gca;
    end

    grid(ax, "on");
    box(ax, "on");
    set(ax, "FontSize", 20, "FontName", "Times");
    set(findall(gcf, "type", "text"), "Color", "k", "FontName", "Times");
    set(gcf, "Position", [100 0 500 500]);
    set(gcf, "PaperOrientation", "landscape");
end

