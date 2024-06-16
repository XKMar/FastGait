#!/usr/bin/env bash
# Available options: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# eg., bash tools/dist_train.sh base FastGait/configs/benchmarks/casia-b_shiftnet_gmpa.yaml

set -x
PYTHON=${PYTHON:-"python"}

METHOD=$1 # base
CONFIG=$2 # the path of config

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
tools/$METHOD/main.py --config=$CONFIG --launcher="pytorch" --tcp-port=${PORT} --set ${PY_ARGS}

# torchrun --nproc_per_node=$GPUS --master_port=$PORT \
# tools/$METHOD/main.py --config=$CONFIG --launcher="pytorch" --tcp-port=${PORT} --set ${PY_ARGS}


