#!/usr/bin/env bash

EXPT_FILE=experiment.txt
NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
MAX_PARALLEL_JOBS=15
sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} --time=06:00:00 --partition=PGR-Standard dispatch_job.sh $EXPT_FILE