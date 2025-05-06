# Code Modified from og_nithin_livefb.py
# Code for generating 4 distortion types and 4 levels for each distortion type for LIVEFB dataset
#
# Last Modified Author: Shika
# Last Modified Date: 7 May 2024
# Created Date: 7 May 2024


import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


def random5k(sample_indicator):
    df = pd.read_csv('./live_fb_split.csv')
    # select only samples where is_valid is False
    df = df[df['is_valid'] == False]
    df = df.sample(n=5000).reset_index(drop=True)
    # we need only mos_image, name_image columns
    df = df[['mos_image', 'name_image']]
    df.index.name = 'index'
    print(df)

    df.to_csv('./LIVEFB_split_%s.csv'%sample_indicator)

    mos = df['mos_image'].to_list()
    plt.hist(mos, bins=20)
    plt.show()

def dist_list(sample_indicator, num_d_levels, num_d_types):
    dist_types = [1, 2, 3, 9, 10, 11, 12, 15, 16, 17, 24, 25]
    dist_levels = [2, 3, 4, 5]
    df = pd.read_csv(f'./LIVEFB_split_{sample_indicator}.csv', index_col='index')
    # print(df)
    df_out = pd.DataFrame(columns=['im_loc', 'ref_loc', 'ref', 'dist_type', 'dist_level'])
    for i in tqdm(range(len(df))):
        ref = '%s.bmp' % df.iloc[i]['im_loc'].split('/')[-1].split('.')[0]
        df_out = df_out.append({'im_loc': '%s_REF.bmp'%ref.split('.')[0],
                                'ref_loc': df.iloc[i]['im_loc'],
                                'ref': ref,
                                'dist_type': 0,
                                'dist_level': 0,
                                }, ignore_index=True)
        d_types = random.sample(dist_types, k=num_d_types)
        for d_type in d_types:
            d_levels = random.sample(dist_levels, k=num_d_levels)
            for d_level in d_levels:
                df_out = df_out.append({'im_loc': '%s_%02d_%02d.bmp'%(ref.split('.')[0], d_type, d_level),
                                        'ref_loc': df.iloc[i]['im_loc'],
                                        'ref': ref,
                                        'dist_type': d_type,
                                        'dist_level': d_level,
                                        }, ignore_index=True)
    
    df_out.index.name = 'index'
    print(df_out)
    df_out.to_csv('./gen/LIVEFB_synthetic.csv')

if __name__ == '__main__':
    
    sample_indicator = '5k'
    num_d_types = 4
    num_d_levels = 4
    dist_list(sample_indicator, num_d_levels, num_d_types)
    # random5k(sample_indicator)
