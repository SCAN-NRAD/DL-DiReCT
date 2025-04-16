import argparse
import os
from multiprocessing.pool import Pool
import numpy as np
import nibabel as nib
from nibabel.processing import conform
import trimesh
from scipy import ndimage
import csv
import vtk
import vtk.util.numpy_support as np_support
from skimage import morphology

def marchingcubes(binary, affine, filename, ns):
    # numpy to vtk array
    vtk_label_arr = np_support.numpy_to_vtk(num_array=binary.ravel(), deep=True, array_type=vtk.VTK_INT)
    vtk_img_data = vtk.vtkImageData()
    vtk_img_data.SetDimensions(binary.shape[2], binary.shape[1], binary.shape[0])
    vtk_img_data.GetPointData().SetScalars(vtk_label_arr)
    vtk_img_prod = vtk.vtkTrivialProducer()
    vtk_img_prod.SetOutput(vtk_img_data)

    # marching cubes
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(vtk_img_prod.GetOutputPort())
    surf.SetValue(0, 1)
    surf.Update()
            
    #smoothing the mesh
    smoother= vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(surf.GetOutputPort())       
    smoother.SetNumberOfIterations(int(ns)) 
    smoother.Update()
    
    # apply the affine transformation
    mat2transf = vtk.vtkMatrixToLinearTransform()
    mat = vtk.vtkMatrix4x4()
    mat.DeepCopy(affine.flatten())
    mat2transf.SetInput(mat)
    mat2transf.Update()    
    transf = vtk.vtkTransformPolyDataFilter()
    transf.SetInputConnection(smoother.GetOutputPort())
    transf.SetTransform(mat2transf)
    transf.Update()

    # Trick to get the mesh vert,faces
    # save temporary output
    # TO DO: get it from vtk.vtkTriangleFilter
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(transf.GetOutputPort())
    writer.SetFileTypeToASCII()            
    writer.SetFileName(filename)
    writer.Write()

def voxealize_mesh(mri_img, mesh, pitch=1, edge_factor=2, max_iter=10):
    # to keep in the img voxel space
    voxealized = np.zeros_like((mri_img))
    
    # remesh until all edges < max_edge (ensure vertex are inside voxel)    
    max_edge = pitch/edge_factor
    v, f, idx = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=max_iter, return_index=True)

    # convert the vertices to their voxel grid position
    hit = np.round(v / pitch).astype(int)

    # remove duplicates
    unique, _inverse = trimesh.grouping.unique_rows(hit)

    # get the voxel centers in model space
    occupied_index = np.array(hit[unique],dtype=np.int32)

    # get the voxealization
    voxealized[tuple(occupied_index.T)] = 1
    # fill the volume
    voxealized_full = ndimage.binary_fill_holes(voxealized)    

    return voxealized_full

def close_voxealization(matrix, radius, ni):
    # close the surface to obtain the smooth pial
#    size = int(radius*2+3)
#    x0, y0, z0 = ((size-1)/2, (size-1)/2, (size-1)/2)
#    x, y, z = np.mgrid[0:size:1, 0:size:1, 0:size:1]
#    r = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
#    r[r > radius] = 0
#    r[r > 0] = 1
#    r[int((size-1)/2), int((size-1)/2), int((size-1)/2)] = 1    
#    structure=r
#    closed = ndimage.binary_closing(matrix,structure=structure,iterations=ni).astype(int)
    closed = morphology.isotropic_closing(matrix,radius+0.5)
    return closed

def extract_smooth_pial(input_list):
    # unpack input
    filepath = input_list[0]
    args = input_list[1]    
    radius = int(args.r)
    ns = int(args.ns)                  
    subj = filepath.split("/")[-1]
    
    hemi = input_list[2]
    
    try:
        # Reading FreeSurfer's reconstruction
        mri_img =  nib.load(filepath+'/mri/raw_brain.mgz')
        pialv, pialf =  nib.freesurfer.io.read_geometry(filepath+'/surf/'+hemi+'h.pial', read_metadata=False, read_stamp=False)
        # get affine
        vox2rastkr = mri_img.header.get_vox2ras_tkr()
        rastkr2vox = np.linalg.inv(vox2rastkr)
                        
        # create a mesh of the pial surface (voxel space)
        mesh_pial = trimesh.Trimesh(nib.affines.apply_affine(rastkr2vox, pialv),pialf)
            
        # mesh voxealized            
        matrix = voxealize_mesh(mri_img.get_fdata(),mesh_pial)
                        
        # close voxalization to obtain the smooth pial
        closed = close_voxealization(matrix, radius, 1)
            
        # create pythonic mesh
        # Trick to get vertex and faces from vtk            
        if not os.path.exists('temp'):
            os.makedirs('temp')                    
        temp_filename = 'temp/{}h_{}.stl'.format(hemi,subj)

        # transpose for the vtk ordering
        marchingcubes(closed.transpose((2,1,0)),vox2rastkr,temp_filename,ns)
        py_smooth_mesh = trimesh.exchange.load.load_mesh(temp_filename)
        trimesh.smoothing.filter_laplacian(py_smooth_mesh, lamb=0.5, iterations=50, implicit_time_integration=False, volume_constraint=True, laplacian_operator=None)
            
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        nib.freesurfer.io.write_geometry(filepath+'/surf/'+hemi+'h.pial-outer-smoothed', py_smooth_mesh.vertices, py_smooth_mesh.faces, create_stamp=None, volume_info=None)

    except FileNotFoundError:
        print("WARNING: Missing reconstruction from subject {}".format(subj), flush = True)

    except KeyboardInterrupt:
        print("The programm was terminated manually!")
        raise SystemExit                                       

# Parser for shell script
parser = argparse.ArgumentParser(prog='smooth_pial_v0', description='This code extracts computes the smooth pial from a Freesurfer reconstruction')
parser.add_argument('-filepath', '--path', help='path to the folder containing the Freesurfer reconstructions for one or multiple subjects', required = True)
parser.add_argument('-nsmooth', '--ns', default = int(500), help='Smoothing steps for the created mesh', required = False)
parser.add_argument('-radius', '--r', default = int(5), help='Radius used to close the surface', required = False)
args = parser.parse_args()

fs_files_path = args.path
hemisphere = ['l','r']

input_list = [(fs_files_path, args, i) for i in hemisphere]

with Pool(2) as pool:
   res = pool.map(extract_smooth_pial, input_list)
