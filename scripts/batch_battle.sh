#!/bin/bash

# export code_dirs="f1 f2 f3 f4 v15"
export code_dirs="f3"
export exps="e3"
for code_dir in $code_dirs
do
    export CODE_DIR=/aiarena/backup_model/yxh_dir/code_battle/${code_dir}
    for exp_name in $exps
    do
        exp="${code_dir}${exp_name}"
        echo "handling $exp"
        CKPT_TGZ=/aiarena/backup_model/yxh_dir/exps/${exp}.tgz
        if [ -e "${CKPT_TGZ}" ]; then
            cd /aiarena/backup_model/yxh_dir/exps
            tar zxvf ${exp}.tgz
            cd /aiarena
        fi
        export CKPT_DIR=/aiarena/backup_model/yxh_dir/exps/${exp}/ckpt
        if [ ! -d "${CKPT_DIR}" ]; then
            continue
        fi
        export LOG_DIR=/aiarena/backup_model/yxh_dir/exps/${exp}/battle
        bash /aiarena/code/scripts/battle.sh
    done
done