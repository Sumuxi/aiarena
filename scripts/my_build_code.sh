#!/bin/bash

# usage: build_code.sh
# env: OUTPUT_DIR, default is ./build
# env: OUTPUT_FILENAME, default is code-$version.zip
# env: MODEL_FILE, model file path

version=2.3.3-$(date +"%Y%m%d%H%M")
filename=code-$version.zip

# current shell script directory
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &>/dev/null && pwd)
ROOT_DIR=$(dirname $SCRIPT_DIR)/
TMP_DIR=$(mktemp -d)

output_dir=${OUTPUT_DIR:-"$SCRIPT_DIR/../build"}
mkdir -p $output_dir
#output_dir=$(cd -- "$output_dir" &>/dev/null && pwd)
filename=${OUTPUT_FILENAME:-"code-$version.zip"}


# build code

# 集群训练不打包learner的init model以减小代码包大小
rsync -a --exclude="checkpoints_*" \
    --exclude="**/checkpoints" \
    --exclude="**/checkpoint" \
    --exclude="**/checkpoint" \
    --exclude="GameAiMgr_*.txt" \
    --exclude="log" \
    --exclude="code/actor/onnx" \
    --exclude="code/assets" \
    --exclude="code/learner" \
    --exclude="code/validate" \
    $ROOT_DIR/code $TMP_DIR

# 复制模型文件
cp $MODEL_FILE $TMP_DIR/code/actor/model/init/model.pth
cp -r $ROOT_DIR/scripts $TMP_DIR

# generate version
echo "$version" >$TMP_DIR/version

# 打包
cd $TMP_DIR && zip -r $output_dir/$filename * && cd -
touch $output_dir/${filename}.done
echo $output_dir/${filename} done

rm -r $TMP_DIR
