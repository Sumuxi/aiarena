# 使用说明

## 软件依赖

```
pip3 install -r requirements.txt
```

## 数据依赖

```
data.bin
output.android.bin
```

## onnx 输出 logits

```
python3 dump_logits.py --data_bin data.bin --model_path ./model/ --output_bin output.onnx.bin
```

> 示例predictor的模型文件放置路径为 `model/3v3.onnx`

## vcap 输出

```
# 安装apk
adb install app-release.apk

# 拷贝文件到手机
adb push data/data.bin /storage/emulated/0/tmp/data.bin
adb push model/ /storage/emulated/0/tmp/model/
adb push lib/ /storage/emulated/0/tmp/lib/

# 启动测试
adb shell pm grant com.vivo.performancetestdemo android.permission.WRITE_EXTERNAL_STORAGE
adb shell pm grant com.vivo.performancetestdemo android.permission.READ_EXTERNAL_STORAGE
adb shell appops set com.vivo.performancetestdemo MANAGE_EXTERNAL_STORAGE allow
adb shell mkdir -p /storage/emulated/0/tmp/output/
adb shell chmod 777 /storage/emulated/0/tmp/output/
adb shell am start -n com.vivo.performancetestdemo/.MainActivity --ez run_test true

# 拉取结果
adb pull /storage/emulated/0/tmp/output/ ./
```


## 对比输出结果

```
python3 validate.py --output_bin_0 output.onnx.bin --output_bin_1 output.vcap.bin
```
