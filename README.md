# UNet-Transformer SIDD 图像去噪项目

本项目使用 `UNet + Transformer` 混合模型完成 sRGB 图像去噪。模型底层结构保持不变：输入 noisy 图像，网络输出噪声残差，再通过

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
- Benchmark 输入：`image/SIDD_Benchmark_Data`
- Benchmark block 坐标：`image/SIDD_Benchmark_Code_v1.2/BenchmarkBlocks32.mat`

`SIDDMediumDataset` 会递归查找 `*_NOISY_SRGB_*.PNG`，并匹配同目录下对应的 `*_GT_SRGB_*.PNG`。

## 训练

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

## 推理与验证抽样

对单张真实 noisy 图像去噪：

```bash
python infer.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --input image.png --output denoised.png
```

如果有 GT，可以同时计算 PSNR：

```bash
python infer.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --input noisy.png --target gt.png --output denoised.png
```

不传 `--input` 时，脚本会从 SIDD 验证 `.mat` 中随机抽样展示 noisy / denoised / GT：

```bash
python infer.py --checkpoint checkpoints/sidd/unet_transformer_best.pth --num-samples 3
```

## 生成 SIDD Benchmark 提交文件

训练完成后运行：

```bash
python submit_sidd.py --checkpoint checkpoints/sidd/unet_transformer_best.pth
```

默认输出：

```text
image/Submit/SubmitSrgb.mat
```

该文件包含官方 sRGB benchmark 需要的 `DenoisedBlocksSrgb`、`TimeMPSrgb` 和 `OptionalData`，可用于提交到 SIDD benchmark 页面。

## 当前实现要点

- 不再在线合成高斯噪声。
- 训练目标为真实噪声残差：`target_noise = noisy - clean`。
- 验证使用官方 SIDD validation blocks。
- 模型结构文件 `src/ut_project/models/unet_transformer.py` 与底层 block 设计保持不变。
