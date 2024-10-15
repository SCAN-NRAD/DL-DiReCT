#!/bin/bash

usage() {
cat << EOF
Usage: batch-dl+direct [-h] [-c n_cpu] [-g n_GPU] [-b] [-m model_file] SRC_DIR DST_DIR
Batch processing using DL+DiReCT (with --bet), with N parallel jobs on CPU and GPU.
SRC_DIR is a directory with data to process, each subject should be in a separate subdirectory (with a T1.nii.gz inside).
Results are written to DST_DIR.

optional arguments:
	-h|--help		show this usage
	-t|--t1			name of T1 nifti (default: T1.nii.gz)
	-c|--cpu		number of parallel CPU jobs (default 8)
	-g|--gpu		number of parallel GPU jobs (default 1)
	-b|--bet		Skull-stripping using hd-bet
	-m|--model		Use given trained model (default v0, and v6 for _ce images)
	
EOF
	exit 0
}

invalid() {
	echo "ERROR: Invalid argument $1"
	usage 1
}

die() {
	echo "ERROR: $1"
	exit 1
}


# defaults
T1_FILE=T1.nii.gz
N_PARALLEL_GPU=1
N_PARALLEL_CPU=8
MODEL_ARGS=""
BET_ARGS=""

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
	case "${1}" in
		-h|--help)	usage 0;;
		-t|--t1)	shift; T1_FILE=$1 ;;
		-g|--gpu)	shift; N_PARALLEL_GPU=$1 ;;
		-c|--cpu)	shift; N_PARALLEL_CPU=$1 ;;
		-b|--bet)	BET_ARGS=" --bet" ;;
		-m|--model)	shift; MODEL_ARGS=" --model $1" ;;
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

export SRC=$1
export DST=$2
export T1_FILE
export BET_ARGS
export MODEL_ARGS

[[ "`which parallel`X" != "X" ]] || die "'parallel' not found. Install it with 'sudo apt install parallel'"
[[ -d ${SRC} ]] || die "Source directory ${SRC} not found"

export JOB_QUEUE=job_queue.txt
[[ -f ${JOB_QUEUE} ]] && rm ${JOB_QUEUE}

TAIL=tail
[[ "`uname -s`" == "Darwin" ]] && TAIL=gtail

run_dl() {
	SUBJ=$1
	START=`date +%s`

	DIR=${DST}/${SUBJ}
	mkdir -p ${DIR}
	if [ ! -f ${DIR}/T1w_norm_seg.nii.gz ] ; then
		if [ -f ${SRC}/${SUBJ}/T1_INV2.nii.gz ] ; then
			# MP2Rage
			BET_ARGS=" ${BET_ARGS} --mp2rage-inv2 ${SRC}/${SUBJ}/T1_INV2.nii.gz"
		fi
		if [[ "${SUBJ}" =~ .*"_ce"$ ]]; then
			# if the subject name ends with '_ce', treat as contrast-enhanced (ce) image
			MODEL_ARGS=" --model v6"
		fi
		dl+direct --subject ${SUBJ} ${BET_ARGS} --no-cth --keep ${MODEL_ARGS} ${SRC}/${SUBJ}/${T1_FILE} ${DIR} 2>&1 >> ${DIR}/dl.log
	fi
	
	STOP=`date +%s`
	DT=$((${STOP}-${START}))

	echo "GPU done (dt=${DT}s): ${SUBJ}"

	echo ${SUBJ} >> ${JOB_QUEUE}
}
export -f run_dl

run_direct() {
	SUBJ=$1
	if [ ${SUBJ} == "dummy" ] ; then
		# ignore 'dummy' entry
		return
	fi

	START=`date +%s`
	DIR=${DST}/${SUBJ}
	if [ ! -f ${DIR}/T1w_norm_thickmap.nii.gz ] ; then
		direct ${SUBJ} ${DIR} 2>&1 >> ${DIR}/direct.log
	fi

	STOP=`date +%s`
	DT=$((${STOP}-${START}))

	echo "CPU done (dt=${DT}s): ${SUBJ}"
}
export -f run_direct

ls ${SRC} | parallel -j ${N_PARALLEL_GPU} run_dl {} &
PID_DL=$!

true > ${JOB_QUEUE} 
# create first N_PARALLEL_CPU dummy entries. Otherwise jobs will only start once N_PARALLEL_CPU jobs are queued
for i in `seq 1 ${N_PARALLEL_CPU}` ; do echo dummy >> ${JOB_QUEUE} ; done

${TAIL} -n+0 -f ${JOB_QUEUE} --pid ${PID_DL} | parallel -j ${N_PARALLEL_CPU} run_direct {}

