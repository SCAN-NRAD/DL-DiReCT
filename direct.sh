#!/bin/bash

usage() {
cat << EOF
Usage: direct SUBJECT_ID OUTPUT_DIR
	Continue DL+DiReCT starting from an existing segmentation in OUTPUT_DIR	
EOF
	exit 0
}

die() {
	echo "ERROR: $1"
	exit 1
}

# defaults
if [ -z "${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS}" ] ; then
	# number of threads for DiReCT
	export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
fi

if [ $# -lt 2 ] ; then
	usage
fi

SUBJECT_ID=$1
DST=$2
SCRIPT_DIR=`dirname $0`/src

[[ -f ${DST}/softmax_seg.nii.gz ]] || die "Invalid OUTPUT_DIR '${DST}': no segmentation found. Did 'dl+direct.sh --no-cth' run?"


# restore input parameters
IN_VOLUME=${DST}/T1w_norm.nii.gz
MASK_VOLUME=${IN_VOLUME}
if [ -f ${DST}/T1w_norm_noskull_mask.nii.gz ] ; then
	MASK_VOLUME=${DST}/T1w_norm_noskull_mask.nii.gz
fi

# DiReCT
python ${SCRIPT_DIR}/DiReCT.py ${DST} ${DST}

# extract stats
THICK_VOLUME=${DST}/T1w_thickmap.nii.gz
python ${SCRIPT_DIR}/extract_stats.py ${THICK_VOLUME} ${DST}/seg.nii.gz ${DST}/softmax_seg.nii.gz ${SUBJECT_ID}

# uncrop to original size
python ${SCRIPT_DIR}/crop.py --revert 1 ${MASK_VOLUME} ${THICK_VOLUME} ${DST}/T1w_norm_thickmap.nii.gz
FS_ARGS=" ${DST}/T1w_norm_thickmap.nii.gz:colormap=heat"

echo
echo "Done, you may view the results with:"
echo -e "\tfreeview ${DST}/T1w_norm.nii.gz ${DST}/T1w_norm_seg.nii.gz:colormap=lut ${FS_ARGS}"
echo


