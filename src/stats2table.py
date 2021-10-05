import pandas as pd
import argparse
import glob
import os
import re


REGEX_SUBCORT = '^(?!lh-|rh-).*'
REGEX_LH = '(SUBJECT.*)|(^lh.*)'
REGEX_RH = '(SUBJECT.*)|(^rh.*)'


def collect_stats(directory, filename):
    df = None
    for f in sorted(glob.glob('{}/*/{}'.format(directory, filename))):
        df_subj = pd.read_csv(f, dtype=str)
        df = df_subj if df is None else df.append(df_subj)
        
    return df


def write_results(df, pattern, dst, suffix=''):
    print('Writing {} rows to {}'.format(len(df), dst))
    df_subset = df.filter(regex=pattern, axis=1)
    # rename to match FS convention (lh- --> lh_ and add _suffix)
    df_subset = df_subset.rename(columns=lambda x: re.sub('(lh|rh)-', r'\1_',x) + (suffix if not x.startswith('SUBJECT') else ''))
    df_subset.to_csv(dst, sep='\t', na_rep='NaN', index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect results from processed subjects and generate FreeSurfer alike stats2table files.')

    parser.add_argument(
        'subjects_dir',
        type=str,
        help='Directory containing processed subjects with DL+DiReCT results.'
    )
    
    parser.add_argument(
        'destination',
        type=str,
        help='Target directory (e.g. stats2table) for generated stats files.'
    )

    args = parser.parse_args()
    if not os.path.exists(args.destination):
        os.makedirs(args.destination, exist_ok=True)
    
    volumes = collect_stats(args.subjects_dir, 'result-vol.csv')
    if volumes is not None:
        write_results(volumes, REGEX_LH, '{}/lh.aparc_stats_volume.txt'.format(args.destination), '_volume')
        write_results(volumes, REGEX_RH, '{}/rh.aparc_stats_volume.txt'.format(args.destination), '_volume')
        
        # add cortex vol as sum of parcellations
        volumes['lhCortexVol'] = volumes.filter(regex=REGEX_LH, axis=1).drop('SUBJECT', axis=1).astype(float).sum(axis=1).astype(int)
        volumes['rhCortexVol'] = volumes.filter(regex=REGEX_RH, axis=1).drop('SUBJECT', axis=1).astype(float).sum(axis=1).astype(int)
        volumes['TotalCortexVol'] = volumes['lhCortexVol'] + volumes['rhCortexVol']
        write_results(volumes, REGEX_SUBCORT, '{}/aseg_stats_volume.txt'.format(args.destination))
        
    
    thick = collect_stats(args.subjects_dir, 'result-thick.csv')
    if thick is not None:
        write_results(thick, REGEX_LH, '{}/lh.aparc_stats_thickness.txt'.format(args.destination), '_thickness')
        write_results(thick, REGEX_RH, '{}/rh.aparc_stats_thickness.txt'.format(args.destination), '_thickness')
    
    thickstd = collect_stats(args.subjects_dir, 'result-thickstd.csv')
    if thickstd is not None:
        write_results(thickstd, REGEX_LH, '{}/lh.aparc_stats_thicknessstd.txt'.format(args.destination), '_thicknessstd')
        write_results(thickstd, REGEX_RH, '{}/rh.aparc_stats_thicknessstd.txt'.format(args.destination), '_thicknessstd')
    