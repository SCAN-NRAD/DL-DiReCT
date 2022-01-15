import argparse
import os
import sys
import radiomics
import SimpleITK as sitk
import csv
import pandas as pd


LABELS_FS = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle',
             '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'Left-VentralDC', 'Left-choroid-plexus',
             'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',
             'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Right-choroid-plexus', '5th-Ventricle',
             'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior']

LABELS_DL = ['Left-Ventricle-all:101', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Left-Hippocampus', 'Left-Amygdala',
             'Left-Accumbens-area', 'Left-VentralDC', 'Right-Ventricle-all:112', 'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
             'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Brain-Stem',
             '3rd-Ventricle', '4th-Ventricle', 'Corpus-Callosum:125']


def lut_parse():
    lut = pd.read_csv('{}/fs_lut.csv'.format(os.path.dirname(os.path.realpath(sys.argv[0]))))
    lut =  dict(zip(lut.Key, lut.Label))
    return lut


def run_main(subject_dirs, aseg_file, labels, results_csv):
    LUT = lut_parse()
    print(results_csv)
    with open(results_csv, 'w') as out_file:
        writer = csv.writer(out_file, delimiter=',')
        header = None
        for subjects_dir in subject_dirs:
            for subject_name in sorted(os.listdir(subjects_dir)):
                fname = '{}/{}/{}'.format(subjects_dir, subject_name, aseg_file)
                if not os.path.exists(fname):
                    print('{}: {} not found. Skipping'.format(subject_name, aseg_file))
                    continue

                print(subject_name)
                fields = list()
                values = list()
                img = sitk.ReadImage(fname)
                for label in labels:
                    if ':' in label:
                        label, label_id = label.split(':')
                    else:
                        label_id = LUT[label]
                    radiomics.setVerbosity(50)
                    shape_features = radiomics.shape.RadiomicsShape(img, img, **{'label': int(label_id)})
                    shape_features.enableAllFeatures()
                    results = shape_features.execute()
                    for key in results.keys():
                        fields.append('{}.{}'.format(label, key))
                        values.append(float(results[key]) if results['VoxelVolume'] > 0 else 'nan')

                if header is None:
                    header = fields
                    writer.writerow(['Subject'] + header)
                else:
                    assert header == fields

                writer.writerow([subject_name] + values)


def main():
    parser = argparse.ArgumentParser(description='Extract radiomics features from subjects')

    parser.add_argument(
        '--aseg_file',
        type=str,
        default='T1w_norm_seg.nii.gz',
        help='Path (relative to subject dir) of aseg segmentation file.'
    )

    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        metavar='label',
        default=['DL'],
        help='List of labels. FreeSurfer ids (from fs_lut) are used per default. '
             'Can also be: label:id. Example: "Left-Hippocampus:9 Right-Hippocampus:21." '
             'Use "FS" for all FreeSurfer labels or "DL" for all DL+DiReCT labels'
    )

    parser.add_argument(
        '--results_csv',
        type=str,
        required=True,
        help='CSV-File to store results'
    )

    parser.add_argument(
        'subject_dirs',
        metavar='dir',
        type=str,
        nargs='+',
        help='Directories with subjects (FreeSurfer or DL+DiReCT results dir)'
    )

    args = parser.parse_args()
    for dir in args.subject_dirs:
        if not os.path.exists(dir):
            print('{} not found'.format(args.subjects_dir))
            sys.exit(1)

    labels = LABELS_FS if args.labels[0] == 'FS' else LABELS_DL if args.labels[0] == 'DL' else args.labels
    run_main(args.subject_dirs, args.aseg_file, labels, args.results_csv)


if __name__ == '__main__':
    sys.exit(main())
