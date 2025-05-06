%% setup
clear; clc;
addpath(genpath('code_imdistort'));
addpath(genpath('dataset_files'));


%% read the info of pristine images

tbl = readtable('LIVEFB_synthetic_temp.csv');
tb = table2cell(tbl);
inpath = '/media/suhas/Data/Suhas/01-ilvqa/Databases/LIVE_FB/';
files = dir(inpath);
outf = '/media/suhas/Data/Suhas/01-ilvqa/Databases/LIVE_FB_syn_18_dist/';
if ~isfolder(outf)
    mkdir(outf);
end
% setColHeading(tbl, 1, 'name');
%% generate distorted images in dist_imgs folder
ref_name = 'asd';
for i = progress(1:len(tb))
% for i = 1:len(tb)
    if ~strcmp(ref_name, tb{i,3})
        ref_im = imread(strcat(inpath,tb{i,3}));
        ref_name = tb{i,3};
        if isa(ref_im,'uint8')~=1
            disp([num2str(i) ' ' class(ref_im)])
        end
        if len(size(ref_im)) == 2
            ref_im = cat(3, ref_im, ref_im, ref_im);
%             disp([num2str(len(size(ref_im))) ' ' ref_name])
        elseif len(size(ref_im)) == 3
            sz = size(ref_im);
            num_ch = sz(3);
            if num_ch ~= 3
                disp([num_ch 'ch ' ref_name])
            end
        else
            disp([len(size(ref_im)) ' ' ref_name])
        end
%         continue
    else
%         continue
    end
    dist_type = tb{i,5};
    dist_level = tb{i,6};
    tempfolder = strcat(outf,tb{i,4},'/');
    if ~isfolder(tempfolder)
        mkdir(tempfolder);
    end
    if dist_level == 0
        imwrite(ref_im, [tempfolder tb{i,2}]);
    else
        [dist_im] = imdist_generator(ref_im, dist_type, dist_level);
        imwrite(dist_im, [tempfolder tb{i,2}]);
    end
end







