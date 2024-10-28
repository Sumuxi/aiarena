#!/bin/bash

# 输入参数：
# CODE_DIR: 代码包路径
# CKPT_DIR: ckpt文件夹路径
# LOG_DIR: 输出的日志路径

echo "code dir: ${CODE_DIR}"
echo "ckpt dir: ${CKPT_DIR}"
echo "log dir: ${LOG_DIR}"

# 对战共 battle_number * concurrency 局
battle_number=120
concurrency=5

# 列出所有 .pth 文件并排序
pth_files=($(ls -1 "$CKPT_DIR"/*.pth 2>/dev/null | sort -r))

# 判断目录是否存在，如果不存在，则创建目录
if [ ! -d "${LOG_DIR}" ]; then
    mkdir "${LOG_DIR}"
fi

for pth_file in "${pth_files[@]}"; do
    echo "Testing model: ${pth_file}"

    # 获得文件名
    # 使用 basename 去掉路径，然后使用参数扩展去掉后缀
    file_name_without_extension="${pth_file##*/}"
    file_name="${file_name_without_extension%.*}"
    # echo "Testing model: ${file_name}"

    log_file="${LOG_DIR}/${file_name}_battle$((battle_number * concurrency)).log"
    echo "log file: ${log_file}"

    if [ -e "${log_file}" ]; then
        echo "skip, because exists log file."
        continue
    fi

    # 把ckpt文件拷贝到code/actor/model/init目录
    echo "Copying ckpt file to initialization directory"
    cp "${pth_file}" "${CODE_DIR}/code/actor/model/init/model.pth" || { echo "Copy failed"; exit 1; }


    # 获取当前时间的时间戳
    start_time=$(date +%s)

    # 运行对战脚本
    echo "Running battle script, please wait..."
    (python3 /workspace/code/backup_model/yxh_dir/baseline/battle/battle.py \
            --driver_0 local_dir \
            --server_0 "${CODE_DIR}" \
            --driver_1 local_dir \
            --server_1 /workspace/code/backup_model/yxh_dir/baseline \
            -n "${battle_number}" -c "${concurrency}" \
            --camp_swap \
            --task_id exp_vs_bs \
            > "${log_file}" 2>&1
    ) &

    # 等待对战进程启动
    sleep 30

    while true; do
        # 检查对战脚本进程是否存在
        if ! pgrep -f "python3 /workspace/code/backup_model/yxh_dir/baseline/battle/battle.py" > /dev/null; then
            break
        fi
        # 获取当前时间的时间戳
        current_time=$(date +%s)
        # 计算运行时间
        runtime=$((current_time - start_time))
        # 如果运行时间超过一定时间，杀掉对战脚本进程
        # if [ $runtime -gt $((battle_number * concurrency * 60)) ]; then
        if [ $runtime -gt 33800 ]; then
            echo "Battle script runs too long, killing the process..."
            pkill -f "python3 /workspace/code/backup_model/yxh_dir/baseline/battle/battle.py"
            break
        fi
        sleep 10
    done

    # 杀掉通过`ps aux | grep actor`获取的所有进程
    echo "Killing all actor processes..."
    ps aux | grep "actor/server.py" | awk '{print $2}' | xargs kill

done