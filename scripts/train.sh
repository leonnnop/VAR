#!/usr/bin/env bash

# Find free port for distributed training
while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        echo "$PORT is free to use";
        break
    fi
done

GPUS=$1
model_name=$2

echo "---------------------------------------------------------"
echo ">>>>>>>> Running training on VAR dataset"
max_n_len=12
max_t_len=22  # including "BOS" and "EOS"
max_v_len=50

data_dir="./data/VAR"
glove_path=${data_dir}"/vocab_feature/"
glove_version="min3_vocab_glove_6B.pt"
echo "---------------------------------------------------------"

PY_CMD="python -m torch.distributed.launch --use_env --nproc_per_node"

${PY_CMD}=${GPUS} --master_port ${PORT} src/runner.py \
--dset_name VAR \
--model_name ${model_name} \
--data_dir ${data_dir} \
--glove_path ${glove_path} \
--glove_version ${glove_version} \
--max_n_len ${max_n_len} \
--max_t_len ${max_t_len} \
--max_v_len ${max_v_len} \
${@:2} 
