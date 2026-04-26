function out = npzload(npzPath)
%NPZLOAD Load a NumPy .npz archive into a MATLAB struct.
%   out = NPZLOAD(npzPath) returns a struct whose fields are the keys of the
%   .npz file. Requires Python with NumPy available to MATLAB.
%
%   This function prefers MATLAB's Python interface (py.*). If Python is not
%   configured, set it via:
%       pyenv("Version","/path/to/python3");
%
%   Tested with .npz files produced by this repo's run_simulations.py.

    arguments
        npzPath (1,:) char
    end

    if ~isfile(npzPath)
        error("npzload:FileNotFound", "File not found: %s", npzPath);
    end

    try
        np = py.importlib.import_module("numpy");
    catch ME
        error("npzload:PythonNumpyMissing", ...
            "Could not import numpy from MATLAB's Python. Configure Python via pyenv and ensure numpy is installed.\nOriginal error:\n%s", ...
            ME.message);
    end

    data = np.load(npzPath, pyargs("allow_pickle", true));

    % data.files can be a py.list (no tolist) or an array-like with tolist.
    if isa(data.files, "py.list") || isa(data.files, "py.tuple")
        keys = cell(data.files);
    else
        try
            keys = cell(data.files.tolist());
        catch
            keys = cell(data.files);
        end
    end

    out = struct();
    for i = 1:numel(keys)
        k = string(keys{i});
        v = data{char(k)};
        out.(matlab.lang.makeValidName(k)) = local_py_to_mat(v);
    end
end

function m = local_py_to_mat(v)
    % Scalars
    if isa(v, "py.numpy.int64") || isa(v, "py.numpy.int32") || isa(v, "py.numpy.float64") || isa(v, "py.float")
        m = double(v);
        return;
    end

    % Python str
    if isa(v, "py.str")
        m = char(v);
        return;
    end

    % NumPy arrays
    if isa(v, "py.numpy.ndarray")
        dt = string(char(v.dtype.name));

        % String arrays (e.g., schemes)
        if startsWith(dt, "str") || contains(dt, "U") || contains(dt, "unicode")
            c = cell(v.tolist());
            m = cellfun(@(x) string(char(x)), c, "UniformOutput", false);
            return;
        end

        % Numeric arrays
        shp = int64(cellfun(@int64, cell(v.shape)));
        flat = v.reshape(int64(-1)).tolist();
        flatCell = cell(flat);
        m = double(cellfun(@(x) double(x), flatCell));
        if numel(shp) == 0
            return;
        end
        % NumPy ndarrays are row-major (C-order). MATLAB reshapes in column-major,
        % so we must compensate to preserve axis meaning.
        if numel(shp) == 1
            m = reshape(m, [double(shp(1)), 1]);
        else
            shp_m = double(fliplr(shp(:).'));           % e.g., (S,nX,nSeeds)->(nSeeds,nX,S)
            m = reshape(m, shp_m);
            m = permute(m, numel(shp):-1:1);            % back to (S,nX,nSeeds)
        end
        return;
    end

    % Fallback for lists/tuples
    if isa(v, "py.list") || isa(v, "py.tuple")
        c = cell(v);
        m = cellfun(@local_py_to_mat, c, "UniformOutput", false);
        return;
    end

    % Unknown: try best-effort conversion
    try
        m = double(v);
    catch
        m = v;
    end
end

