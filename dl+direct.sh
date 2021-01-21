#!/bin/bash

usage() {
cat << EOF
Usage: dl+direct [OPTION]... T1_FILE OUTPUT_DIR
Process T1_FILE (.nii.gz) with dl+direct and put results into OUTPUT_DIR.
Input is expected to be a skull-stripped T1w MRI. You may specify --bet to remove
the skull (using hd-bet).

	-h|--help	show this usage
	-s|--subject	subject-id (written to .csv results)
	-b|--bet	Skull-stripping using hd-bet
	-n|--no-cth	Skip cortical thickness (DiReCT), just perform segmentation	
	-m|--model	Use given trained model
	-k|--keep	Keep intermediate files
	
EOF
	exit 0
}

invalid() {
	echo "ERROR: Invalid argument $1"
	usage 1
}

die() {
	echo "$1"
	exit 1
}

# defaults
SUBJECT_ID="subj_id"
DO_SKULLSTRIP=0
DO_CT=1
KEEP_INTERMEDIATE=0
MODEL_ARGS=""
if [ -z "${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS}" ] ; then
	# number of threads for DiReCT
	export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
fi

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
	case "${1}" in
		-h|--help)	usage 0;;
		-s|--subject)	shift; SUBJECT_ID="$1" ;;
		-b|--bet)	DO_SKULLSTRIP=1 ;;
		-n|--no-cth)	DO_CT=0 ;;
		-m|--model)	shift; MODEL_ARGS="--model $1" ;;
		-k|--keep)	KEEP_INTERMEDIATE=1 ;;
		-*)		invalid "$1" ;;
		*)		POSITIONAL+=("$1") ;;
	esac
	shift
done

# Restore positional parameters
set -- "${POSITIONAL[@]}"

if [ $# -lt 2 ] ; then
	usage 1
fi

T1=$1
DST=$2
SCRIPT_DIR=`dirname $0`/src

# check prerequisites
[[ -f ${T1} ]] || die "Invalid input volume: ${T1} not found"
[[ ${DO_SKULLSTRIP} -eq 0 ]] || [[ "`which hd-bet`X" != "X" ]] || die "hd-bet not found. Install it from https://github.com/NeuroAI-HD/HD-BET"

mkdir -p ${DST} || die "Could not create target directory ${DST}"

echo
echo "If you are using DL+DiReCT in your research, please cite:"
cat ${SCRIPT_DIR}/../doc/cite.md
echo

# convert into freesurfer space (resample to 1mm voxel, orient to LIA)
python ${SCRIPT_DIR}/conform.py ${T1} ${DST}/T1w_norm.nii.gz

HAS_GPU=`python -c 'import torch; print(torch.cuda.is_available())'`
BET_OPTS=""
if [ ${HAS_GPU} != 'True' ] ; then
	echo "Warning: No GPU/CUDA device found. Running on CPU might take some time... (running hd-bet in fast mode, check results!)"
	BET_OPTS=" -device cpu -mode fast -tta 0 "
fi


# Skull-stripping
if [ ${DO_SKULLSTRIP} -gt 0 ] ; then
	# skull-strip using HD-BET
	hd-bet -i ${DST}/T1w_norm.nii.gz -o ${DST}/T1w_norm_noskull.nii.gz ${BET_OPTS} || die "hd-bet failed"
	IN_VOLUME=${DST}/T1w_norm_noskull.nii.gz
	MASK_VOLUME=${DST}/T1w_norm_noskull_mask.nii.gz
else
	# Assume input is already skull-stripped
	IN_VOLUME=${DST}/T1w_norm.nii.gz
	MASK_VOLUME=${IN_VOLUME}
fi

# cropping
IN_VOLUME_CROP=${DST}/T1w_norm_noskull_cropped.nii.gz
python ${SCRIPT_DIR}/crop.py ${MASK_VOLUME} ${IN_VOLUME} ${IN_VOLUME_CROP}


# DeepScan segmentation
python ${SCRIPT_DIR}/DeepSCAN_Anatomy_Newnet_apply.py ${MODEL_ARGS} ${IN_VOLUME_CROP} ${DST} ${SUBJECT_ID} || die "Segmentation failed"

if [ ${DO_CT} -gt 0 ] ; then
	# DiReCT
	python ${SCRIPT_DIR}/DiReCT.py ${DST} ${DST} || die "DiReCT failed"

	# extract stats
	THICK_VOLUME=${DST}/T1w_thickmap.nii.gz
	python ${SCRIPT_DIR}/extract_stats.py ${THICK_VOLUME} ${DST}/seg.nii.gz ${DST}/softmax_seg.nii.gz ${SUBJECT_ID}
fi

FS_ARGS=""
# uncrop to original size
if [ ${DO_CT} -gt 0 ] ; then
	python ${SCRIPT_DIR}/crop.py --revert 1 ${MASK_VOLUME} ${THICK_VOLUME} ${DST}/T1w_norm_thickmap.nii.gz
	FS_ARGS=" ${DST}/T1w_norm_thickmap.nii.gz:colormap=heat"
fi
python ${SCRIPT_DIR}/crop.py --revert 1 ${MASK_VOLUME} ${DST}/softmax_seg.nii.gz ${DST}/T1w_norm_seg.nii.gz

# cleanup
if [ ${KEEP_INTERMEDIATE} -eq 0 ] ; then
	rm -f ${DST}/{boundary*.nii.gz,?mprob*.nii.gz,seg.nii.gz,seg_*.nii.gz,softmax_seg.nii.gz,T1w_norm_noskull_*.nii.gz,T1w_thickmap.nii.gz}
fi

echo
echo "Done, you may view the results with:"
echo -e "\tfreeview ${DST}/T1w_norm.nii.gz ${DST}/T1w_norm_seg.nii.gz:colormap=lut ${FS_ARGS}"
echo


