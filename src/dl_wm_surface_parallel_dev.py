# Reconstruction of a topollogically correct WM surface
# based on DL+DiReCT segmentation
#
# DeepSCAN: a deep learning-based neuroanatomy segmentation and cortex parcellation 
# DiReCT: A Diffeomorphic registration based cortical thickness
#
# For instructions to produce a Segmentation: https://github.com/SCAN-NRAD/DL-DiReCT
#
# this code produces for each hemisphere: The white matter surface
# 
# Victor B. B. Mello, 09/2024
# Support Center for Advanced Neuroimaging (SCAN)
# University Institute of Diagnostic and Interventional Neuroradiology
# University of Bern, Inselspital, Bern University Hospital, Bern, Switzerland.

import argparse
import nibabel as nib
import numpy as np
import pandas as pd
import nighres
import pymeshlab
from nibabel.processing import conform
from multiprocessing.pool import Pool

# For FS visualization
def get_vox2ras_tkr(t1):
    # Transformation for FreeView visualization
    ds = t1.header._structarr['pixdim'][1:4]
    ns = t1.header._structarr['dim'][1:4] * ds / 2.0
    v2rtkr = np.array([[-ds[0], 0, 0, ns[0]],
                       [0, 0, ds[2], -ns[2]],
                       [0, -ds[1], 0, ns[1]],
                       [0, 0, 0, 1]], dtype=np.float32)
                       
    return v2rtkr
    
def rec_surf(binary, affine,r):
    # Reconstruct topologically correct surfaces
    # https://nighres.readthedocs.io/en/latest/shape/topology_correction.html
    # https://nighres.readthedocs.io/en/latest/surface/levelset_to_mesh.html
    # Ref: Bazin and Pham (2007). Topology correction of segmented medical images using a fast marching algorithm doi:10.1016/j.cmpb.2007.08.006
    farray_img = nib.Nifti1Image(binary.astype(np.float64), affine)

    propag = 'background->object'
    connect = '6/18'
    minimum_distance = 1e-5
                    
    ret = nighres.shape.topology_correction(farray_img, 'binary_object', minimum_distance=minimum_distance, propagation=propag,connectivity=connect)
    l2m_ret = nighres.surface.levelset_to_mesh(ret['corrected'], connectivity=connect)
    vertices = l2m_ret['result']['points']
    faces = l2m_ret['result']['faces']
        
    return vertices, faces
    
def run_rec(input_list):
    seg_img = input_list[0]
    df_labels = input_list[1]
    affine = input_list[2]
    region = input_list[3]
    
    # Regions based on dl output label_def.csv
    if region == 'lh':
        labels = [df_labels['ID'][x] for x in df_labels['ID'].keys() if (x.startswith('Left') and x != 'Left-Cerebellum') ]
        mask = np.isin(seg_img, labels)
        binary = np.array(np.where(mask,1,0),dtype=np.int32)

    elif region == 'rh':
        labels = [df_labels['ID'][x] for x in df_labels['ID'].keys() if (x.startswith('Right') and x != 'Right-Cerebellum') ]
        mask = np.isin(seg_img, labels)
        binary = np.array(np.where(mask,1,0),dtype=np.int32)        
    else:
        print('Not a valid region!')
        raise SystemExit
    
    # Reconstruct a topologically correct surface
    vertices, faces = rec_surf(binary, affine, region)

    # apply affine for FS visualization and matching with the MRI
    transf_vertx = nib.affines.apply_affine(affine,vertices)
    # get a smooth mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=transf_vertx,face_matrix=faces), region)
    ms.meshing_invert_face_orientation()
    ms.apply_coord_taubin_smoothing(stepsmoothnum=nsmooth)
        
    m=ms.current_mesh()
    # FreeSurfer binary files
    # Visualization together with T1w_norm.nii.gz        
    nib.freesurfer.io.write_geometry(args.output+'/surf/'+region+'.white', m.vertex_matrix(), m.face_matrix(), create_stamp=None, volume_info=None)
    nib.freesurfer.io.write_geometry(args.output+'/surf/'+region+'.orig', m.vertex_matrix(), m.face_matrix(), create_stamp=None, volume_info=None)
    nib.freesurfer.io.write_geometry(args.output+'/surf/'+region+'.white.preaparc', m.vertex_matrix(), m.face_matrix(), create_stamp=None, volume_info=None)

    # create annotation with unknown labels
    # will be changed in the future        
    names = [b'unknown']
    ctab = np.array([[25, 5, 25, 0,  1639705]], dtype=np.int32)
    annot = np.array( np.zeros(len(m.vertex_matrix())), dtype = np.int32)
    nib.freesurfer.io.write_annot(args.output+'/label/'+region+'.aparc.annot', annot, ctab, names, fill_ctab=True)    
        
# Parser for shell script
parser = argparse.ArgumentParser()
parser.add_argument('-inputpath', '--input')
parser.add_argument('-outputpath', '--output')
parser.add_argument('-ns', '--nsmooth')
args = parser.parse_args()
nsmooth = int(args.nsmooth)

# Read DL+DiReCT results
# segmentation
seg_img = nib.load(args.input+'/mri/aparc.atlas+aseg.nii.gz')
affine = get_vox2ras_tkr(seg_img)
seg = seg_img.get_fdata()
df_labels = pd.read_csv(args.input+'/label_def.csv').set_index('LABEL').to_dict()

# reconstruct different ROIs
regions = ['lh','rh']
input_list = [(seg,df_labels,affine,i) for i in regions]

with Pool() as pool:
    res = pool.map(run_rec, input_list)

