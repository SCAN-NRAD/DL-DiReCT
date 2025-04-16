#!/bin/bash
SCRIPT_DIR=$1
SUBJECT_ID=$2
DST=$3
MODEL_ARGS=$4

echo ${MODEL_ARGS}

FS_VERSION_SUPPORT="7.4.1"
if [ -z "$FREESURFER_HOME" ]
  then
  echo "  ERROR: Did not find \$FREESURFER_HOME. A working version of FreeSurfer $FS_VERSION_SUPPORT"
  echo "  is needed to run the surface reconstruction."
else
  if grep -q "${FS_VERSION_SUPPORT}" "$FREESURFER_HOME/build-stamp.txt"
    then
    echo "Doing the fast surface reconstruction based on Freesurfer"

    mkdir -p ${DST}/mri
    mkdir -p ${DST}/label
    mkdir -p ${DST}/surf

    # conform to LIA 256x256x256
    # transform CC label to WM label in hemisphere basis
    # change Pial labels to GM in a hemisphere basis aseg.mgz
    # create filled.mgz and wm.seg
    python ${SCRIPT_DIR}/preparedata.py -inputpath ${DST}

    mri_normalize -aseg ${DST}/mri/aseg.presurf.mgz ${DST}/mri/norm.mgz ${DST}/mri/brain.mgz 
    cp ${DST}/mri/brain.mgz ${DST}/mri/norm.mgz
    cp ${DST}/mri/brain.mgz ${DST}/mri/nu.mgz
    cp ${DST}/mri/brain.mgz ${DST}/mri/brain.finalsurfs.mgz

    mri_edit_wm_with_aseg ${DST}/mri/wm.seg.mgz ${DST}/mri/brain.mgz ${DST}/mri/aseg.presurf.mgz ${DST}/mri/wm.asegedit.mgz 
    mri_pretess ${DST}/mri/wm.asegedit.mgz wm ${DST}/mri/brain.mgz ${DST}/mri/wm.mgz  

    python ${SCRIPT_DIR}/dl_wm_surface_parallel_dev.py -inputpath ${DST} -outputpath ${DST} -ns 50

    # Produce surfaces and labes based on segmentation
    export SUBJECTS_DIR=${DST}/..
    FOLDER_NAME=$(basename ${DST})
    parallel mris_make_surfaces -orig_white white.preaparc -orig_pial white.preaparc -aseg aseg.presurf -nowhite -mgz -T1 brain.finalsurfs ${FOLDER_NAME} {} ::: lh rh

    # smooth outer pial surface
    python ${SCRIPT_DIR}/smooth_pial.py -filepath ${DST}
    
    # change the filenames to .raw
    # indicate that some post processing is done to meet Freesurfer
    mv ${DST}/surf/lh.pial ${DST}/surf/lh.pial.raw
    mv ${DST}/surf/rh.pial ${DST}/surf/rh.pial.raw

    mv ${DST}/surf/lh.thickness ${DST}/surf/lh.thickness.raw
    mv ${DST}/surf/rh.thickness ${DST}/surf/rh.thickness.raw
    
    python ${SCRIPT_DIR}/extract_morphometrics.py -filepath ${DST} -subj ${SUBJECT_ID}
    
    if [ "'${MODEL_ARGS}'" == "'--model v7'" ];
    then
      echo "yes"
      mv ${DST}/label/lh.aparc.annot ${DST}/label/lh.aparc.a2009s.annot
      mv ${DST}/label/rh.aparc.annot ${DST}/label/rh.aparc.a2009s.annot
    else
      echo "${MODEL_ARGS}"      
    fi
    
    cp ${DST}/mri/raw_brain.mgz ${DST}/mri/orig.mgz
     
    echo "Surfaces Done!"
  else
    echo " Conversion, software version $FS_VERSION_SUPPORT. "
    echo " ERROR: You are trying to run the surface reconstruction with FreeSurfer version $(cat "$FREESURFER_HOME/build-stamp.txt"). The code was tested only with FreeSurfer $FS_VERSION_SUPPORT."
  fi
fi
