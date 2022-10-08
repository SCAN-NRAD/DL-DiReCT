import pandas as pd
import numpy as np
import scipy.stats as stats
import nibabel as nib
import os
import sys
import csv
import skimage.segmentation as segmentation
import scipy.spatial as spatial
import argparse


# get cortical labels
def get_labels(offset):
    lut = pd.read_csv('{}/fs_lut.csv'.format(os.path.dirname(os.path.realpath(sys.argv[0]))))
    lut = {l.Key: l.Label for _, l in lut.iterrows() if l.Label >= offset and l.Label <= 3000+offset}
    labels = list(filter(lambda x: ((x.startswith('lh') or x.startswith('rh')) and not 'nknown' in x and not 'corpuscallosum' in x and not 'Medial_wall' in x), lut.keys()))
    all_labels = labels + ['lh-MeanThickness', 'rh-MeanThickness']
    
    return lut, labels, all_labels


# get average and std thickness per parcellation (averaging over non-zero voxels in each given parcellation)
def get_stats(thickness_map, parcellation):
    # DK or destrieux atlas
    offset = 0 if np.max(parcellation) <= 3000 else 10000
    lut, labels, all_labels = get_labels(offset)
    
    rois = [thickness_map[np.where(parcellation == lut[lbl])] for lbl in labels]
    # lh/rh MeanThickness
    rois = rois + [
		thickness_map[np.where(np.logical_and(parcellation >= 1000+offset, parcellation < 2000+offset))],
		thickness_map[np.where(np.logical_and(parcellation >= 2000+offset, parcellation < 3000+offset))]
		]
    return np.array([roi[roi.nonzero()].mean() for roi in rois]), np.array([roi[roi.nonzero()].std() for roi in rois]), all_labels


def write_stats(stats, subject_id, fname, label_names):    
    with open(fname, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['SUBJECT'] + label_names)
        writer.writerow([subject_id] + list(stats))
		

def save_img(img, fname, ref_img):
    niftiImg = nib.Nifti1Image(img, ref_img.affine)
    niftiImg.header['xyzt_units'] = 2  # mm
    nib.save(niftiImg, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract thickness stats')
    parser.add_argument("thick_map")
    parser.add_argument("seg_file")
    parser.add_argument("aparc_file")
    parser.add_argument("subject_id")
    args = parser.parse_args()
    
    thick_map = args.thick_map
    seg_file = args.seg_file
    aparc_file = args.aparc_file
    subject_id = args.subject_id
    dst_dir = os.path.dirname(thick_map)
    
    thickness_map_nib = nib.load(thick_map)
    
    thickness_map = thickness_map_nib.get_fdata(dtype=np.float32)
    ref_img = thickness_map_nib
    parcellation = nib.load(aparc_file).get_fdata(dtype=np.float32)
    
    # average over boundary only    
    seg_img = nib.load('{}'.format(seg_file)).get_fdata(dtype=np.float32)
    seg_gm = np.zeros_like(seg_img).astype(np.uint8)
    seg_gm[seg_img == 2] = 1
	
    #outer_boundary = segmentation.find_boundaries(seg_wm, connectivity=1, mode='outer').astype(np.uint8)
    # inner GM boundary
    inner_boundary = segmentation.find_boundaries(seg_gm, connectivity=1, mode='inner').astype(np.uint8)
    save_img(inner_boundary, '{}/boundary_def.nii.gz'.format(dst_dir), ref_img)
	
    # build KDTree with coordinates of all parcellation voxels
    coords_parc = np.array(np.where(parcellation > 1000)).T
    tree = spatial.cKDTree(coords_parc)
    
    # identify coordinates of boundary voxels
    coords_boundary = np.array(np.where(inner_boundary > 0)).T
    
    # for every boundary voxel, find nearest parcellation voxel
    nearest_distances, nearest_indices = tree.query(coords_boundary, k=1)
    nearest_coords = coords_parc[nearest_indices]
    nearest_labels = parcellation[nearest_coords[:, 0], nearest_coords[:, 1], nearest_coords[:, 2]]
    
    # for every boundary voxel, assign label of closest parcellation
    parc_nearest = np.zeros_like(inner_boundary).astype(np.uint16)
    parc_nearest[coords_boundary[:, 0], coords_boundary[:, 1], coords_boundary[:, 2]] = nearest_labels
    
    # up to now, also boundary voxel in e.g. the hippocampus got a label (from the closest parcellation)
    # remove those, i.e. mask > sqrt(3) euclidean dist
    indices_to_mask = np.array(np.where(nearest_distances > np.sqrt(3.0))).T
    print('Masking {} voxels'.format(indices_to_mask.shape[0]))
    coords_to_mask = coords_boundary[indices_to_mask[:, 0]]
    parc_nearest[coords_to_mask[:, 0], coords_to_mask[:, 1], coords_to_mask[:, 2]] = 0
    save_img(parc_nearest, '{}/boundary_nearest_masked.nii.gz'.format(dst_dir), ref_img)
    
    stats_nearest_mean, stats_nearest_std, label_names = get_stats(thickness_map, parc_nearest)
    print('mean thick: {}'.format(np.mean(stats_nearest_mean[-2:])))
    write_stats(stats_nearest_mean, subject_id, '{}/result-thick.csv'.format(dst_dir), label_names)
    write_stats(stats_nearest_std, subject_id, '{}/result-thickstd.csv'.format(dst_dir), label_names)
    
    