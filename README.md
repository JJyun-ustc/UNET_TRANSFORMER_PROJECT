# UNet-Transformer SIDD 图像去噪项目

本项目使用 `UNet + Transformer` 混合模型完成 sRGB 图像去噪。通过

```python
denoised = noisy - pred_noise
```

得到去噪图像。当前数据流程已经从“干净图像在线随机加噪”改为 SIDD Medium 的真实噪声图/GT 成对训练。

## 项目结构

```text
unet_transformer_project
├─ train.py
├─ infer.py
├─ submit_sidd.py
├─ submit_sidd_csv.py
├─ requirements.txt
├─ image
│  ├─ train_data
│  │  ├─ ReadMe_sRGB.txt
│  │  ├─ Scene_Instances.txt
│  │  └─ Data
│  ├─ ValidationNoisyBlocksSrgb.mat
│  ├─ ValidationGtBlocksSrgb.mat
│  ├─ SIDD_Benchmark_Data
│  └─ SIDD_Benchmark_Code_v1.2
└─ src/ut_project
   ├─ config.py
   ├─ data/dataset.py
   ├─ engine/trainer.py
   └─ models
```

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括 `torch`、`torchvision`、`matplotlib`、`Pillow` 和 `scipy`。其中 `scipy` 用于读取验证集 `.mat` 与生成 benchmark 提交文件。

## 数据说明

默认使用项目内的 SIDD 路径：

- 训练集：`image/train_data/Data`
- 验证 noisy blocks：`image/ValidationNoisyBlocksSrgb.mat`
- 验证 GT blocks：`image/ValidationGtBlocksSrgb.mat`
- Kaggle Benchmark 输入块：`BenchmarkNoisyBlocksSrgb.mat`
- MATLAB Benchmark 输入：`image/SIDD_Benchmark_Data`
- MATLAB Benchmark block 坐标：`image/SIDD_Benchmark_Code_v1.2/BenchmarkBlocks32.mat`

`SIDDMediumDataset` 会递归查找 `*_NOISY_SRGB_*.PNG`，并匹配同目录下对应的 `*_GT_SRGB_*.PNG`。

## 训练

第一次运行前建议先确认数据都在默认位置：

```text
image/train_data/Data
image/ValidationNoisyBlocksSrgb.mat
image/ValidationGtBlocksSrgb.mat
image/SIDD_Benchmark_Data
```

开始训练：

```bash
python train.py
```

常用参数：

```bash
python train.py --epochs 50 --batch-size 8 --patch-size 256 --pairs-per-image 4
```

如需自定义数据路径：

```bash
python train.py ^
  --train-root image/train_data/Data ^
  --val-noisy-mat image/ValidationNoisyBlocksSrgb.mat ^
  --val-gt-mat image/ValidationGtBlocksSrgb.mat
```

训练会保存：

- `checkpoints/sidd/unet_transformer_latest.pth`
- `checkpoints/sidd/unet_transformer_best.pth`

其中 best checkpoint 按验证集 PSNR 保存。

## 验证与推理

### 全量验证

`infer.py` 默认使用 SIDD validation blocks。SIDD validation 通常包含 40 组图像块，每组 32 个 block，共 1280 个验证块。

一次跑完整验证集并打印指标：

```bash
python infer.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --validate-all
```

同时保存逐块指标 CSV：

```bash
python infer.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --validate-all --metrics-output outputs/sidd_val_metrics.csv
```

输出指标包括：

- `validated blocks`：验证块数量
- `avg_psnr`：平均 PSNR
- `min_psnr`：最低 PSNR
- `max_psnr`：最高 PSNR
- `avg_mse`：平均 MSE

### 抽样可视化

不传 `--input` 且不加 `--validate-all` 时，脚本会从 SIDD 验证 `.mat` 中随机抽样展示 noisy / denoised / GT：

```bash
python infer.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --num-samples 3
```

### 单图推理

对单张真实 noisy 图像去噪：

```bash
python infer.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --input image.png --output denoised.png
```

如果有 GT，可以同时计算 PSNR：

```bash
python infer.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --input noisy.png --target gt.png --output denoised.png
```

## 生成 SIDD Benchmark 提交文件

### Kaggle CSV 提交

官方 Kaggle 提交流程要求读取 `BenchmarkNoisyBlocksSrgb.mat`，输出 `SubmitSrgb.csv`。本项目对应脚本是：

```bash
python submit_sidd_csv.py --checkpoint checkpoints/sidd/unet_transformer_best.pth
```

如果当前目录没有 `BenchmarkNoisyBlocksSrgb.mat`，脚本会尝试从官方 URL 下载：

```text
http://130.63.97.225/share/SIDD_Blocks/BenchmarkNoisyBlocksSrgb.mat
```

也可以手动下载后指定路径：

```bash
python submit_sidd_csv.py ^
  --checkpoint checkpoints/sidd/unet_transformer_best.pth ^
  --input-file BenchmarkNoisyBlocksSrgb.mat ^
  --output SubmitSrgb.csv
```

输出文件：

```text
SubmitSrgb.csv
```

该 CSV 包含两列：

- `ID`：block 编号
- `BLOCK`：去噪结果的 base64 编码

提交位置：

```text
kaggle.com/competitions/sidd-benchmark-srgb-psnr
```

可以先少量 block 做烟雾测试：

```bash
python submit_sidd_csv.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --limit-blocks 4 --output SubmitSrgb_test.csv
```

正式提交不要使用 `--limit-blocks`。

### 旧 MATLAB MAT 提交

如果需要兼容旧版 MATLAB benchmark 代码，可以继续生成 `SubmitSrgb.mat`：

```bash
python submit_sidd.py --checkpoint checkpoints/sidd/unet_transformer_best.pth
```

默认输出：

```text
image/Submit/SubmitSrgb.mat
```

该文件包含官方 sRGB benchmark 需要的 `DenoisedBlocksSrgb`、`TimeMPSrgb` 和 `OptionalData`，可用于提交到 SIDD benchmark 页面。

## 推荐完整流程

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 训练模型：

```bash
python train.py --epochs 50 --batch-size 8 --patch-size 256 --pairs-per-image 4
```

3. 全量验证并保存指标：

```bash
python infer.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --validate-all --metrics-output outputs/sidd_val_metrics.csv
```

4. 抽样查看视觉效果：

```bash
python infer.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --num-samples 3
```

5. 生成 Kaggle benchmark 提交文件：

```bash
python submit_sidd_csv.py --checkpoint checkpoints/sidd/unet_transformer_best.pth
```

6. 提交 `SubmitSrgb.csv` 到 Kaggle SIDD benchmark。
