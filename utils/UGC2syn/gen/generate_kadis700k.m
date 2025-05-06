%% setup
clear; clc;
addpath(genpath('code_imdistort'));


%% read the info of pristine images

tbl = readtable('kadis700k_ref_imgs.csv');
tb = table2cell(tbl);
inpath = '../part_10_refs/';
files = dir(inpath);
outf = './dist_imgs/';
mkdir(outf);
% setColHeading(tbl, 1, 'name');
%% generate distorted images in dist_imgs folder

for i = 3:size(files)
          image_name = files(i).name;
%         try
          ref_im = imread(strcat(inpath,image_name));
          iRow = find(strcmp(tbl.ref_im,image_name)==1);
          
          dist_type = tb{iRow,2};
          tempfolder = strcat(outf,image_name,'/');
          mkdir(tempfolder);   
          imwrite(ref_im, [tempfolder image_name(1:end-4) '_REF.png']);
          for dist_level = 1:5
              [dist_im] = imdist_generator(ref_im, dist_type, dist_level);
              strs = split(image_name,'.');
              dist_im_name = [strs{1}  '_' num2str(tb{iRow,2},'%02d')  '_' num2str(dist_level,'%02d') '.bmp'];

              imwrite(dist_im, [tempfolder dist_im_name]);
          end

        disp(i);
    
    
    
end







