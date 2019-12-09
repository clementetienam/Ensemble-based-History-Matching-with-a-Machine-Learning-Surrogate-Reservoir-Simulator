function out = read_ecl(filename)
% READ_ECL  Read binary files from Schlumberger's reservoir simulator, Eclipse
%
%    out = read_ecl('filename') returns a struct containing all data from
%    the binary file specified by 'filename'. 
%
%    Example:
%    
%    Let 'FIVESPOT.UNRST' be a unified, binary restart file from a typical
%    simple Eclipse run. Then the command
%
%      out = read_ecl('FIVESPOT.UNRST')
%
%    produces a struct with several fields, e.g., SEQNUM, INTEHEAD, 
%    LOGIHEAD, DOUBHEAD, IGRP, SGRP, XGRP, ZGRP, IWEL, SWEL, XWEL, ZWLS, 
%    IWLS, ICON, SCON, XCON, DLYTIM, STARTSOL, PRESSURE, SWAT, ENDSOL. Of
%    these, PRESSURE and SWAT are grid data for the pressure and water
%    saturation, respectively, with one column for every report step. For a 
%    cartesian grid with horizontal dimensions nx * ny, saturation for the
%    first report step at the topmost layer can be plotted using the
%    following code snippet
%
%      report_step = 1;
%      top_layer = reshape(out.SWAT(1:nx*ny, report_step), nx, ny);
%      scale = [0 1];
%      imagesc(top_layer, scale);
%
%    Author: Pål Næverlid Sævik


    out = struct;

    if ~exist('filename', 'var')
        error(['''' filename ''' does not exist']); end

    % Open file
    fclose all;
    fid = fopen(filename);
    if fid < 3; error 'Error while opening file'; end
    
    % Skip header
    fread(fid, 1, 'int32=>double', 0, 'b');
    
    % Read one property at the time
    i = 0;
    while ~feof(fid)
        i = i + 1;
        
        % Read field name (keyword) and array size
        keyword = deblank(fread(fid, 8, 'uint8=>char')');
        keyword = strrep(keyword, '+', '_');
        num = fread(fid, 1, 'int32=>double', 0, 'b');
        
        % Read and interpret data type
        dtype = fread(fid, 4, 'uint8=>char')';
        switch dtype
            case 'INTE'
                conv =  'int32=>double';
                wsize = 4;
            case 'REAL'
                conv = 'single=>double';
                wsize = 4;
            case 'DOUB'
                conv = 'double';
                wsize = 8;
            case 'LOGI'
                conv = 'int32';
                wsize = 4;
            case 'CHAR'
                conv = 'uint8=>char';
                num = num * 8;
                wsize = 1;
        end
        
        % Skip next word
        fread(fid, 1, 'int32=>double', 0, 'b');
        
        % Read data array, which may be split into several consecutive
        % arrays
        data = [];
        remnum = num;
        while remnum > 0
            % Read array size
            buflen = fread(fid, 1, 'int32=>double', 0, 'b');
            bufnum = buflen / wsize;
            
            % Read data and append to array
            data = [data; fread(fid, bufnum, conv, 0, 'b')]; %#ok<AGROW>
            
            % Skip next word and reduce counter
            fread(fid, 1, 'int32=>double', 0, 'b');
            remnum = remnum - bufnum;
        end
        
        % Special post-processing of the LOGI and CHAR datatypes
        switch dtype
            case 'LOGI'
                data = logical(data);
            case 'CHAR'
                data = reshape(data, 8, [])';
        end

        % Add array to struct. If keyword already exists, append data.
        if isfield(out, keyword)
            out.(keyword) = [out.(keyword), data];
        else
            out.(keyword) = data;
        end

        % Skip next word
        fread(fid, 1, 'int32=>double', 0, 'b');
    end
        
    fclose(fid);
end