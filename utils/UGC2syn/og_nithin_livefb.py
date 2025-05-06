import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

sample_indicator = '5k'

def random2k():
    df = pd.read_csv('./dataset_files/LIVEFB.csv', index_col=['index'])
    df = df.sample(n=5000).reset_index(drop=True)
    df.index.name = 'index'
    print(df)

    df.to_csv('./dataset_files/LIVEFB_%s.csv'%sample_indicator)

    mos = df['mos'].to_list()
    plt.hist(mos, bins=20)
    plt.show()

def dist_list():
    dist_types = [1, 2, 3, 9, 10, 11, 12, 15, 16, 17, 24, 25]
    dist_levels = [2, 3, 4, 5]
    df = pd.read_csv('./LIVEFB_%s.csv'%sample_indicator, index_col='index')
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
        d_types = random.sample(dist_types, k=4)
        for d_type in d_types:
            d_levels = random.sample(dist_levels, k=2)
            for d_level in d_levels:
                df_out = df_out.append({'im_loc': '%s_%02d_%02d.bmp'%(ref.split('.')[0], d_type, d_level),
                                        'ref_loc': df.iloc[i]['im_loc'],
                                        'ref': ref,
                                        'dist_type': d_type,
                                        'dist_level': d_level,
                                        }, ignore_index=True)
        
        # if i > 5:
        #     break
    
    df_out.index.name = 'index'
    print(df_out)
    df_out.to_csv('./LIVEFB_synthetic.csv')


def dist_list_v2():
    dist_types = [1, 2, 3, 9, 10, 11, 12, 15, 16, 17, 24, 25]
    dist_levels = [1, 2, 3, 4, 5]
    # df = pd.read_csv('./dataset_files/LIVEFB_%s.csv'%sample_indicator, index_col='index')
    df = pd.read_csv('./dataset_files/LIVEFB.csv', index_col='index')
    # print(df)
    df_out = pd.DataFrame(columns=['im_loc', 'ref_loc', 'ref', 'dist_type', 'dist_level'])
    for i in tqdm(range(len(df))):
        ref = '%s.bmp' % df.iloc[i]['im_loc'].split('/')[-1].split('.')[0]
        d_types = random.sample(dist_types, k=1)
        df_out = df_out.append({'im_loc': '%s_REF.bmp'%ref.split('.')[0],
                                'ref_loc': df.iloc[i]['im_loc'],
                                'ref': ref,
                                'dist_type': d_types[0],
                                'dist_level': 0,
                                }, ignore_index=True)
        
        for d_type in d_types:
            d_levels = dist_levels
            for d_level in d_levels:
                df_out = df_out.append({'im_loc': '%s_%02d_%02d.bmp'%(ref.split('.')[0], d_type, d_level),
                                        'ref_loc': df.iloc[i]['im_loc'],
                                        'ref': ref,
                                        'dist_type': d_type,
                                        'dist_level': d_level,
                                        }, ignore_index=True)
        
        # if i > 5:
        #     break
    
    df_out.index.name = 'index'
    print(df_out)
    df_out.to_csv('./dataset_files/LIVEFB_synthetic_full_v2.csv')


def dist_list_mBlur():
    dist_types = 3
    dist_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # df = pd.read_csv('./dataset_files/LIVEFB_%s.csv'%sample_indicator, index_col='index')
    df = pd.read_csv('./dataset_files/LIVEFB_5k.csv', index_col='index')
    # print(df)
    df_out = pd.DataFrame(columns=['im_loc', 'ref_loc', 'ref', 'dist_type', 'dist_level'])
    for i in tqdm(range(len(df))):
        ref = '%s.bmp' % df.iloc[i]['im_loc'].split('/')[-1].split('.')[0]
        df_out = df_out.append({'im_loc': '%s_REF.bmp'%ref.split('.')[0],
                                'ref_loc': df.iloc[i]['im_loc'],
                                'ref': ref,
                                'dist_type': dist_types,
                                'dist_level': 0,
                                }, ignore_index=True)
        
        for d_level in dist_levels:
            df_out = df_out.append({'im_loc': '%s_%02d_%02d.bmp'%(ref.split('.')[0], dist_types, d_level),
                                    'ref_loc': df.iloc[i]['im_loc'],
                                    'ref': ref,
                                    'dist_type': dist_types,
                                    'dist_level': d_level,
                                    }, ignore_index=True)
        
        # if i > 5:
        #     break
    
    df_out.index.name = 'index'
    print(df_out)
    df_out.to_csv('./dataset_files/LIVEFB_mBlur.csv')



if __name__ == '__main__':
    # random2k()
    dist_list()
    # dist_list_v2()
    # dist_list_mBlur()
