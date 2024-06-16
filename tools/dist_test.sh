#!/usr/bin/env bash
# Available options: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# eg., bash tools/dist_test.sh records/models/CASIA_B_cut_64_pkl/base/ 40

set -x

PYTHON=${PYTHON:-"python"}

WORKDIR=$1
EPOCH=$2
PY_ARGS=${@:3}

GPUS=${GPUS:-8}

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
tools/test.py --work-dir=${WORKDIR} --epoch=${EPOCH} --launcher="pytorch" --tcp-port=${PORT} --set ${PY_ARGS}
