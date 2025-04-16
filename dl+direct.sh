#!/bin/bash

usage() {
cat << EOF
Usage: dl+direct [-h] [-s subject] [-b [-i inv2_file]] [-n] [-f] [-g] [-m model_file] [-k] T1_FILE OUTPUT_DIR
Process T1_FILE (nifti) with dl+direct and put results into OUTPUT_DIR.
Input is expected to be a skull-stripped T1w MRI. You may specify --bet to remove
the skull (using hd-bet).

optional arguments:
	-h|--help		show this usage
	-s|--subject		subject-id (written to .csv results)
	-b|--bet		Skull-stripping using hd-bet
	-i|--mp2rage-inv2	Use given 2nd inversion recovery image from an MP2Rage to generate brain mask
	-n|--no-cth		Skip cortical thickness (DiReCT)
	-f|--no-fsr             Skip fast surface reconstruction
	-g|--no-seg             Skip segmentation
	-m|--model		Use given trained model
	-k|--keep		Keep intermediate files
	-l|--lowmem		Use less memory (use fp16 for ensembling)
	
EOF
	exit 0
}

invalid() {
	echo "ERROR: Invalid argument $1"
	usage 1
}

die() {
	RET=$?
	echo "ERROR (${RET}): $1"
	if [ ${RET} -eq 137 ] ; then
		 echo "Likely out-of-memory. Try with '--lowmem' option"
	fi
	exit 1
}

# defaults
SUBJECT_ID="subj_id"
DO_SKULLSTRIP=0
DO_CT=1
DO_SEG=1
DO_FSR=1
KEEP_INTERMEDIATE=0
LOW_MEM_ARG=""
MODEL_ARGS=""
MP2RAGE_INV2=""
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
		-i|--mp2rage-inv2)	shift; MP2RAGE_INV2=$1 ;;
		-n|--no-cth)	DO_CT=0 ;;
		-f|--no-fsr)    DO_FSR=0 ;;
		-g|--no-seg)    DO_SEG=0 ;;
		-m|--model)	shift; MODEL_ARGS="--model $1" ;;
		-k|--keep)	KEEP_INTERMEDIATE=1 ;;
		-l|--lowmem)	LOW_MEM_ARG="--lowmem True" ;;
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
[[ ${DO_SKULLSTRIP} -eq 0 ]] || [[ "`which hd-bet`X" != "X" ]] || die "hd-bet not found. Install it from https://github.com/MIC-DKFZ/HD-BET"

mkdir -p ${DST} || die "Could not create target directory ${DST}"

echo
echo "If you are using DL+DiReCT in your research, please cite:"
cat ${SCRIPT_DIR}/../doc/cite.md
echo

# convert into freesurfer space (resample to 1mm voxel, orient to LIA)
python ${SCRIPT_DIR}/conform.py "${T1}" "${DST}/T1w_norm.nii.gz"

HAS_GPU=`python -c 'import torch; print(torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))'`
if [ ${HAS_GPU} != 'True' ] ; then
	echo "WARNING: No GPU/CUDA device found. Running on CPU might take some time..."
fi

# Skull-stripping
if [ ${DO_SKULLSTRIP} -gt 0 ] ; then
	# skull-strip using HD-BET
	BET_OPTS=""
	if [ "${MP2RAGE_INV2}x" != "x" ] ; then
		echo "Using ${MP2RAGE_INV2} to create brain mask"
		BET_INPUT_VOLUME=${DST}/T1w_mp2rage_INV2_norm.nii.gz
		BET_OPTS=" --mp2rage-inv2 ${BET_INPUT_VOLUME}"
		python ${SCRIPT_DIR}/conform.py ${MP2RAGE_INV2} "${BET_INPUT_VOLUME}"
	fi
	IN_VOLUME=${DST}/T1w_norm_noskull.nii.gz
	BET_INPUT_VOLUME=${DST}/T1w_norm.nii.gz
	MASK_VOLUME=${DST}/T1w_norm_noskull_mask.nii.gz
	export PYTORCH_ENABLE_MPS_FALLBACK=1
	
	python ${SCRIPT_DIR}/bet.py ${BET_OPTS} "${BET_INPUT_VOLUME}" "${IN_VOLUME}" || die "hd-bet failed"
else
	# Assume input is already skull-stripped
	IN_VOLUME=${DST}/T1w_norm.nii.gz
	MASK_VOLUME=${IN_VOLUME}
fi

# cropping
IN_VOLUME_CROP=${DST}/T1w_norm_noskull_cropped.nii.gz
python ${SCRIPT_DIR}/crop.py "${MASK_VOLUME}" "${IN_VOLUME}" "${IN_VOLUME_CROP}"

if [ ${DO_SEG} -gt 0 ] ; then
	# DeepScan segmentation
  	python ${SCRIPT_DIR}/DeepSCAN_Anatomy_Newnet_apply.py ${LOW_MEM_ARG} ${MODEL_ARGS} "${IN_VOLUME_CROP}" "${DST}" "${SUBJECT_ID}" || die "Segmentation failed"
fi

if [ ${DO_CT} -gt 0 ] ; then
	# DiReCT
	python ${SCRIPT_DIR}/DiReCT.py "${DST}" "${DST}" || die "DiReCT failed"

	# extract stats
	THICK_VOLUME=${DST}/T1w_thickmap.nii.gz
	python ${SCRIPT_DIR}/extract_stats.py "${THICK_VOLUME}" "${DST}/seg.nii.gz" "${DST}/softmax_seg.nii.gz" "${SUBJECT_ID}"
fi

FS_ARGS=""
# uncrop to original size
if [ ${DO_CT} -gt 0 ] ; then
	python ${SCRIPT_DIR}/crop.py --revert 1 "${MASK_VOLUME}" "${THICK_VOLUME}" "${DST}/T1w_norm_thickmap.nii.gz"
	FS_ARGS=" ${DST}/T1w_norm_thickmap.nii.gz:colormap=heat"
fi
python ${SCRIPT_DIR}/crop.py --revert 1 "${MASK_VOLUME}" "${DST}/softmax_seg.nii.gz" "${DST}/T1w_norm_seg.nii.gz"

if [ ${DO_FSR} -gt 0 ] ; then
    echo "${MODEL_ARGS}"
    fast_surface_reconstruction ${SCRIPT_DIR} ${SUBJECT_ID} ${DST} "'$MODEL_ARGS'"
fi

# cleanup
if [ ${KEEP_INTERMEDIATE} -eq 0 ] ; then
	rm -f ${DST}/{boundary*.nii.gz,?mprob*.nii.gz,seg.nii.gz,seg_*.nii.gz,softmax_seg.nii.gz,T1w_norm_noskull_*.nii.gz,T1w_thickmap.nii.gz}
fi

echo
echo "Done, you may view the results with:"
echo -e "\tfreeview ${DST}/T1w_norm.nii.gz ${DST}/T1w_norm_seg.nii.gz:colormap=lut ${FS_ARGS}"
echo


