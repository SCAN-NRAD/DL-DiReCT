import nibabel as nib
import nibabel.processing as nib_processing
import nibabel.orientations as nib_orientations
import nibabel.funcs as nib_funcs
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reorient to LIA and resample to 1mm iso-voxel resolution if required')

    parser.add_argument(
        'source',
        type=str,
        help='Input volume'
    )
    
    parser.add_argument(
        'destination',
        type=str,
        help='Normalized volume'
    )

    args = parser.parse_args()
    
    src_nib = nib_funcs.squeeze_image(nib.load(args.source))
    current_orientation = ''.join(nib.aff2axcodes(src_nib.affine))
    print('Input: {} [{}]'.format(src_nib.header.get_zooms(), current_orientation))
    
    
    # Avoid resampling if already 1mm iso-voxel
    # Note: Also in cases of tiny rounding error, e.g. (1.0000001, 1.0000001, 1.0)
    if not np.allclose(src_nib.header.get_zooms(), [1, 1, 1]):
        # requires re-sampling
        print('Resampling')
        dst_nib = nib_processing.conform(src_nib, orientation='LIA')
    elif current_orientation != 'LIA':
        # requires just reorient
        print('Reorientating {} to LIA'.format(current_orientation))
        start_ornt = nib_orientations.io_orientation(src_nib.affine)
        end_ornt = nib_orientations.axcodes2ornt('LIA')
        transform = nib_orientations.ornt_transform(start_ornt, end_ornt)
        dst_nib = src_nib.as_reoriented(transform)
    else:
        dst_nib = src_nib
    
    nib.save(dst_nib, args.destination)
    