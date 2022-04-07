import pandas as pd
import numpy as np
import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser(description='Generate FreeSurfer alike stats2table files from radiomics.csv (see radiomics_extractor).')
    
    parser.add_argument(
        '--file_prefix',
        type=str,
        default='aseg_stats_',
        help='Prefix to add to output CSV filename.'
    )

    parser.add_argument(
        'input_csv',
        type=str,
        help='Input CSV with radiomics stats'
    )
    
    parser.add_argument(
        'destination',
        type=str,
        help='Target directory (e.g. stats2table) for generated stats files.'
    )

    args = parser.parse_args()
    if not os.path.exists(args.destination):
        os.makedirs(args.destination, exist_ok=True)
        
        
    df = pd.read_csv(args.input_csv, dtype=str)
    labels = np.unique([c.split('.')[0] if '.' in c else c for c in list(df.columns)[1:]])
    features = np.unique([c.split('.')[1] if '.' in c else c for c in list(df.columns)[1:]])
    print('Found {} features and {} labels for {} subjects'.format(len(features), len(labels), len(df)))
    
    for feature in features:
        df_subset = df.filter(regex='Subject|.*\\.{}$'.format(feature))
        df_subset = df_subset.rename(columns=lambda x: re.sub('\.', '_',x))
        df_subset.to_csv('{}/{}{}.txt'.format(args.destination, args.file_prefix, feature), sep='\t', na_rep='NaN', index=False)


if __name__ == '__main__':
    sys.exit(main())
