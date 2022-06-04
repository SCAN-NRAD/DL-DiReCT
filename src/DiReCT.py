import numpy as np
import scipy.special as scipy_special
import nibabel as nib
import os
import sys
import argparse
import ants


def save_img(img, dst, name, ref_img):
    fname = '{}/{}.nii.gz'.format(dst, name)
    niftiImg = nib.Nifti1Image(img, ref_img.affine)
    niftiImg.header['xyzt_units'] = 2  # mm
    nib.save(niftiImg, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare input for DiReCT')
    parser.add_argument("source_dir")
    parser.add_argument("destination_dir")
    args = parser.parse_args()
    
    src = args.source_dir
    dst = args.destination_dir
    
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    GM_labels = ['Left-Cerebral-Cortex', 'Right-Cerebral-Cortex', 'Left-Amygdala', 'Right-Amygdala', 'Left-Hippocampus', 'Right-Hippocampus']
    WM_labels = ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter'] + (['WM-hypointensities'] if os.path.exists('{}/seg_WM-hypointensities.nii.gz'.format(src)) else [])
    
    # to get affine
    ref_img = nib.load('{}/seg_{}.nii.gz'.format(src, WM_labels[0]))

    gm_logit_all = np.stack(
        [nib.load('{}/seg_{}.nii.gz'.format(src, label)).get_fdata(dtype=np.float32) for label in GM_labels])
    
    wm_logit_all = np.stack(
        [nib.load('{}/seg_{}.nii.gz'.format(src, label)).get_fdata(dtype=np.float32) for label in WM_labels])
    
    gm_logit = np.max(gm_logit_all, axis=0)
    wm_logit = np.max(wm_logit_all, axis=0)
    
    # handle background correctly (must remain zero!!)
    gm_prob = np.where(gm_logit == 0, 0, scipy_special.expit(gm_logit))
    wm_prob = np.where(wm_logit == 0, 0, scipy_special.expit(wm_logit))
    
    # fill holes (especially between GM/WM boundary): use argmax if combined probability > 0.7 but individual below 0.5
    seg_img = (np.argmax([gm_prob, wm_prob], axis=0)+2).astype(np.uint8)
    seg_img[gm_prob+wm_prob < 0.7] = 0
    
    seg_img[gm_prob > 0.5] = 2
    seg_img[wm_prob > 0.5] = 3
    
    
    # (before thresholding)
    save_img(gm_prob, dst, 'gmprob', ref_img)
    save_img(wm_prob, dst, 'wmprob', ref_img)
    
    # thresholding
    # (need to make sure we actually have a gray/white matter interface
    # which is defined as direct neighbouring GMprob > 0.5 with a neighbouring WMprob > 0.5)
    T=0.5
    
    gm_prob[wm_prob > gm_prob] = 0
    wm_prob[wm_prob > gm_prob] = 1
    wm_prob[seg_img < 1] = 0
    gm_prob[gm_prob > T] = 1
    wm_prob[gm_prob > T] = 0
    
    save_img(seg_img, dst, 'seg', ref_img)
    save_img(gm_prob, dst, 'gmprobT', ref_img)
    save_img(wm_prob,dst,  'wmprobT', ref_img)
    
    # run DiReCT (KellyKapowski), equivalent to
    #     KellyKapowski -d 3 -s ${DST}/seg.nii.gz -g ${DST}/gmprobT.nii.gz -w ${DST}/wmprobT.nii.gz -o ${THICK_VOLUME} -c "[ 45,0.0,10 ]" -v
    thick = ants.kelly_kapowski(s=ants.from_numpy(seg_img), g=ants.from_numpy(gm_prob), w=ants.from_numpy(wm_prob), c='[ 45,0.0,10 ]', v='1')
    save_img(thick.numpy(), dst, 'T1w_thickmap', ref_img)
    