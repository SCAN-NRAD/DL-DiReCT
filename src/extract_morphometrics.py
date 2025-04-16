import numpy as np
import nibabel as nib
import pymeshlab
import trimesh
import argparse
import os
import sys
import csv
from multiprocessing.pool import Pool
from scipy import spatial, ndimage
import pandas as pd

def partArea(vertices, faces, condition):
    # Area of a selected region
    # Dealing with the border: 1/3 triangle area if only one vertex is inside the ROI
    # 2/3 triangle area if 2 vertices are inside the ROI
    r_faces = np.isin(faces, condition)    
    fid = np.sum(r_faces,axis=1)
    area = 0
    array_area = trimesh.triangles.area(vertices[faces])
    array_area_corrected = np.zeros_like(array_area)
        
    for i in range(1,4,1):
        area = area + i*np.sum(array_area[fid==i])/3
        array_area_corrected[fid==i] = i*array_area[fid==i]/3

    return area, array_area_corrected
    
def map_voxelvalue(pts_arr, volume):
    # get the voxel value from an array of indexes
    intarray = np.array(pts_arr, dtype=np.int32)
    label_pts = volume[intarray[:,0],intarray[:,1],intarray[:,2]].reshape(len(intarray),1)
    
    return label_pts

def do_convex_hull(vertices, faces):
    ms = pymeshlab.MeshSet()            
    m = pymeshlab.Mesh(vertex_matrix=vertices,face_matrix=faces)
    ms.add_mesh(m,'surface')
    ms.generate_convex_hull()    
    ms.set_current_mesh(1)   

    return ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()

def make_label_facebased(vertices, faces, label):
    # map vertex label into face label
    ms = pymeshlab.MeshSet()            
    m = pymeshlab.Mesh(vertex_matrix=vertices,face_matrix=faces, v_scalar_array=label)
    ms.add_mesh(m,'surface')
    ms.compute_scalar_transfer_vertex_to_face() 

    return ms.current_mesh().face_scalar_array() 
    
def transfer_label(initialv, initialf, scalar_label, finalv, finalf):
    # Transfer labels from mesh to mesh by proximity
    # label with the same label of the closest labeled vertex
    ms_transfer = pymeshlab.MeshSet()
    morig = pymeshlab.Mesh(vertex_matrix=initialv,face_matrix=initialf,v_scalar_array=scalar_label)
    mfinal = pymeshlab.Mesh(vertex_matrix=finalv,face_matrix=finalf)
    ms_transfer.add_mesh(morig,'initial_mesh')
    ms_transfer.add_mesh(mfinal,'final_mesh')        
    ms_transfer.transfer_attributes_per_vertex(sourcemesh = 0, targetmesh = 1, qualitytransfer = True)
    ms_transfer.compute_scalar_transfer_vertex_to_face()        
    ms_transfer.set_current_mesh(1)
    m = ms_transfer.current_mesh()
    
    return m.vertex_scalar_array()

# freesurfer definition of thickness
def get_freesurfer_distance(white_surf, pial_surf):
    # closest distance from white to pial
    tree = spatial.cKDTree(pial_surf.vertices) 
    closest_thickness_pial, idx = tree.query(white_surf.vertices, k=1)

    # from those points calculate the closest distance to white
    tree = spatial.cKDTree(white_surf.vertices) 
    closest_thickness_wm, idx = tree.query(pial_surf.vertices[idx], k=1)
    
    return (closest_thickness_pial + closest_thickness_wm) / 2

# used to match points to the segmentation
def label_points(points, aparc, max_dst=3):

    # build tree for labeling
    coords_parc = np.array(np.where(aparc >= 1000)).T # we limit to cortex, and offset to be at center of voxel
    tree = spatial.cKDTree(coords_parc + 0.5) # offset to be at the center of voxel

    # and label for thickness
    nearest_distances, nearest_indices = tree.query(points, k=1) # we use the wm, it's more stable
    nearest_coords = coords_parc[nearest_indices] 
    nearest_labels = np.array(aparc[nearest_coords[:, 0], nearest_coords[:, 1], nearest_coords[:, 2]], dtype=int)
    nearest_labels[nearest_distances > max_dst] = 0 # contrain distance,
    
    return nearest_labels

def create_annot(labels, lut, dst):
    filtered_lut = lut[lut['Label'].isin(np.unique(labels))].reset_index(drop=True)
    mapping = {v: k for k, v in filtered_lut.Label.items()}
    labels = pd.Series(labels).map(mapping).values
    ctab = filtered_lut[["R", "G", "B", "T", "Label"]].values.astype(int)
    nib.freesurfer.io.write_annot(dst, labels, ctab, [(reg[7:] if reg[:3]=='ctx' else reg) for reg in filtered_lut.Key.values], fill_ctab=True)

def write_stats(stats, subject_id, fname, label_names):    
    with open(fname, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['SUBJECT'] + label_names)
        writer.writerow([subject_id] + list(stats))
		

# Parser for shell script
parser = argparse.ArgumentParser()
parser.add_argument('-filepath', '--path', required = True)
parser.add_argument('-subj', '--id', required = True)
args = parser.parse_args()
filepath = args.path
subjID = args.id

# parcellation
seg_img = nib.load(filepath+'/mri/aparc.atlas+aseg.nii.gz')
seg = seg_img.get_fdata()

# labels
offset = 0 if np.max(seg) <= 3000 else 10000
lut = pd.read_csv('{}/fs_lut.csv'.format(os.path.dirname(os.path.realpath(sys.argv[0]))))
lut = {l.Key: l.Label for _, l in lut.iterrows() if l.Label >= offset and l.Label <= 3000+offset}

out_thickness = {}
out_wm_area = {}
out_pial_area = {}
out_meanK = {}
out_contrast = {}
out_gaussK = {}

hemispheres = ['l','r']
for idx, hemi in enumerate(hemispheres):                          
    region_list = list(filter(lambda x: ((x.startswith(hemi+'h')) and not 'nknown' in x and not 'corpuscallosum' in x and not 'Medial_wall' in x), lut.keys()))    
    region_label = [[lut[reg]] for reg in region_list]
    
    # Pial surface
    pialv, pialf = nib.freesurfer.io.read_geometry(filepath+"/surf/"+hemi+"h.pial.raw")
    pial_surf = trimesh.Trimesh(pialv, pialf)
    pial_surf = trimesh.smoothing.filter_humphrey(pial_surf, iterations=25)        
    # save the pial smoothed
    nib.freesurfer.io.write_geometry(filepath+'/surf/'+hemi+'h.pial', pial_surf.vertices, pial_surf.faces, create_stamp=None, volume_info=None)
                                               
    # WM surface
    whitev, whitef = nib.freesurfer.io.read_geometry(filepath+"/surf/"+hemi+"h.white")                                                            
    white_surf = trimesh.Trimesh(whitev, whitef)
    white_surf_smoothed = trimesh.smoothing.filter_humphrey(white_surf, iterations=50)
    nib.freesurfer.io.write_geometry(filepath+'/surf/'+hemi+'h.white.smoothed', white_surf_smoothed.vertices, white_surf_smoothed.faces, create_stamp=None, volume_info=None)    
    
    # freesurfer thickness definition
    thickness = get_freesurfer_distance(white_surf_smoothed, pial_surf)
    nib.freesurfer.io.write_morph_data(filepath + '/surf/'+hemi+'h.thickness', thickness, fnum=0)
        
    # Pial surface
    smoothpialv, smoothpialf = nib.freesurfer.io.read_geometry(filepath+"/surf/"+hemi+"h.pial-outer-smoothed")

    # Mean Curvature
    mean_k = nib.freesurfer.io.read_morph_data(filepath+"/surf/"+hemi+"h.curv.pial")

    # MRI image
    mri_img = nib.load(filepath+"/mri/raw_brain.mgz")
    intensity_val = mri_img.get_fdata() 
    ras_tkr2vox = np.linalg.inv(mri_img.header.get_vox2ras_tkr())
    voxel_size = mri_img.header.get_zooms()[0]

    # make wm/gm contrast map
    # from FS surface coordinate to voxel space
    whitev_voxel_space = nib.affines.apply_affine(ras_tkr2vox, whitev)

    # label WM (mask hemispheres to avoid mapping contralateral regions)
    masked_seg = np.zeros(seg.shape, dtype='int')
    for label in region_label:
        masked_seg[seg == label] = label
    annot = label_points(whitev_voxel_space, masked_seg)

    # save annotation
    color_lut = pd.read_csv("{}/fs_color.csv".format(os.path.dirname(os.path.realpath(sys.argv[0]))))
    create_annot(annot, color_lut,filepath+'/label/'+hemi+'h.aparc.annot')

    # mean thickness facebased
    t_facebased = make_label_facebased(pialv,pialf,thickness)        
    annot_facebased = make_label_facebased(pialv,pialf,annot)

    # create mesh in the voxel space
    mwhite = pymeshlab.Mesh(vertex_matrix=whitev_voxel_space,face_matrix=whitef)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mwhite,'surf')
    ms.compute_normal_per_face()
    ms.apply_normal_normalization_per_face()
    ms.meshing_invert_face_orientation()        
        
    # Sample voxels
    # WM: Intensity 1 mm inside WM (face normal direction)
    # GM: Intensity 30% of the thickness inside GM (face normal direction)
    # normal pointing outside WM
    white_fnormals = ms.current_mesh().face_normal_matrix()                
    center = np.mean(whitev_voxel_space[whitef],axis=1)                        
    wm = map_voxelvalue(center-white_fnormals,intensity_val)
    gm = map_voxelvalue(center+white_fnormals*t_facebased.reshape(-1,1)*0.3/voxel_size,intensity_val) # consistent with FS definition
    temp_contrast = 2*(wm-gm)/(gm+wm) # consistent with FS definition
    nib.freesurfer.io.write_morph_data(filepath + '/surf/'+hemi+'h.wmgm_contrast', temp_contrast, fnum=0)    
    
    # Topological invariant curvature
    kg = trimesh.curvature.discrete_gaussian_curvature_measure(pial_surf, pial_surf.vertices, 0)        
    nib.freesurfer.io.write_morph_data(filepath + '/surf/'+hemi+'h.gaussian_curvature', temp_contrast, fnum=0)    

    region_list.append(hemi+'h_HemisphereMean')
    all_lbl = list(lut.values())
    region_label.append(all_lbl)
    labels_dict = dict(zip(region_list, region_label))
    for v, k, in labels_dict.items():
        label_selection = np.where(np.isin(annot, k))

        # pial and WM areas
        pial_area, pial_area_array = partArea(pialv, pialf, condition = label_selection)
        white_area, white_area_array = partArea(whitev, whitef, condition = label_selection)            

        # Thickness: weight by triangle area
        w = (pial_area_array/np.sum(pial_area_array) + white_area_array/np.sum(white_area_array))/2
        avg_thickness = np.sum(w*t_facebased)
            
        # integrated Gaussian curvature
        int_kg = np.sum(np.absolute(kg[label_selection]))

        # Mean curvature
        avg_meank = np.sum(np.absolute(mean_k[label_selection]))

        # contrast
        reg_contrast = temp_contrast[label_selection]
        mean_contrast = np.mean(reg_contrast[ reg_contrast > 0 ])
        out_thickness[v] = avg_thickness
        out_wm_area[v] = white_area
        out_pial_area[v] = pial_area
        out_meanK[v] = avg_meank
        out_contrast[v] = mean_contrast
        out_gaussK[v] = int_kg

    out_thickness[hemi+'h_MeanThickness'] = out_thickness[hemi+'h_HemisphereMean']
    del out_thickness[hemi+'h_HemisphereMean']
    out_wm_area[hemi+'h_WhiteSurfArea'] = out_wm_area[hemi+'h_HemisphereMean']
    del out_wm_area[hemi+'h_HemisphereMean']
    del out_gaussK[hemi+'h_HemisphereMean']
    del out_meanK[hemi+'h_HemisphereMean']    
    del out_contrast[hemi+'h_HemisphereMean']    
    
# Export stats
write_stats(list(out_thickness.values()), subjID, '{}/result-SBM-FS-thickness.csv'.format(filepath), list(out_thickness.keys())) # thickness
write_stats(list(out_wm_area.values()), subjID, '{}/result-SBM-WM-area.csv'.format(filepath), list(out_wm_area.keys())) # WM area
#write_stats(list(out_pial_area.values()), subjID, '{}/result-SBM-Pial-area.csv'.format(filepath), list(out_pial_area.keys())) # Pial area
write_stats(list(out_meanK.values()), subjID, '{}/result-SBM-mean-curvature.csv'.format(filepath), list(out_meanK.keys())) # mean curvature
write_stats(list(out_gaussK.values()), subjID, '{}/result-SBM-gauss-curvature.csv'.format(filepath), list(out_gaussK.keys())) # Gaussian curvature
write_stats(list(out_contrast.values()), subjID, '{}/result-SBM-contrast.csv'.format(filepath), list(out_contrast.keys())) # contrast

