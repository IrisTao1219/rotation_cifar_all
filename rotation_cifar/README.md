# 四方向共享特征聚合课程项目

本目录已有训练代码、固定数据划分、完成的检查点、日志、确认集结果和图。推荐实验使用`aug4_large`与`pool_max_large`两个模式，参数量均为256,830。`pool_mean_large`用于聚合方式消融。其他模式保留为选题过程的否定或对照记录。

实验结论：5,000张独立确认图像、三种子，特征最大聚合75.41%，开发集选择的最强简单预测融合73.56%，平均+1.85个百分点。核心思想借鉴TI-pooling，属于课程模块集成与验证。

## 现成服务器项目

位置：`/home/hehaotian/ml-course-newtopic-20261011/rotation_cifar`

已用环境：`/home/hehaotian/ml-course-patchcore/.venv/bin/python`，PyTorch 2.5.1+cu121、NumPy 1.26.4。该共享环境未被修改。详细版本见`environment.json`。

单次80轮训练在本次共享4090环境下约2分钟，运行时间会随同机负载变化。程序把每个训练进程的PyTorch显存分配上限设为显存的20%；模型不需要多卡并行训练。`CUDA_VISIBLE_DEVICES`请选择空闲卡。

已有seed1～3的模型；直接检查现成结果：

```bash
cd /home/hehaotian/ml-course-newtopic-20261011/rotation_cifar
COURSE_PY=/home/hehaotian/ml-course-patchcore/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 "$COURSE_PY" evaluate_fixed.py --run pool_max_large_s1_e80 --split dev
```

新增一次配对训练可使用seed4，避免覆盖已经完成的实验：

```bash
CUDA_VISIBLE_DEVICES=0 "$COURSE_PY" experiment.py --mode aug4_large --seed 4 --epochs 80
CUDA_VISIBLE_DEVICES=0 "$COURSE_PY" experiment.py --mode pool_max_large --seed 4 --epochs 80
```

训练程序禁止覆盖同名结果目录。若要完整复现seed1～3，请在新的项目目录保留同一份`manifest.json`与数据文件后运行，不要删除原有证据。

## 文件说明

| 文件 | 用途 |
|---|---|
| `experiment.py` | 网络、共享四方向编码、最大/平均特征聚合及全部训练对照 |
| `prepare_hf.py` | 从哈希固定的训练parquet准备数据，按原图划分 |
| `manifest.json` | 训练/开发/确认的固定原图ID、数据版本和校验值 |
| `evaluate_fixed.py` | 按开发集选定的检查点评估，确认集需要冻结清单 |
| `aggregate.py` | 三种子统计和以原图为单位的配对bootstrap |
| `strong_control_check.py` | 应用开发集选择的简单融合方式，核验更强对照 |
| `confirmation_lock.json` | 打开确认集前的方案、种子、检查点名单及代码哈希 |
| `runs/` | 配置、逐轮日志、模型权重和逐图预测 |
| `results/` | 可机器读取的最终统计结果 |
| `figures/` | 已检查的结果图、每类结果和成功/失败案例 |
| `选题建议与实验报告.md` | 课题解释、证据、创新边界和文献 |

每个`best.pt`由开发集选择，确认集不参与模型选择。训练默认只读取train和dev；不要为了获得更好数字，反复利用confirm修改方案。

## 从头准备数据

已有`data/train.npz`、`data/dev.npz`、`data/confirm.npz`时可跳过。新的独立环境需要PyTorch、NumPy，预处理另外需要Pillow和PyArrow，绘图需要Matplotlib。安装说明仅针对你新建的环境；无需改动现有共享环境。

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 pillow pyarrow matplotlib
mkdir -p data
curl -L --fail 'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text/train-00000-of-00001.parquet' -o data/train.parquet
python prepare_hf.py
```

训练文件SHA256为`8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883`，脚本会核验。仅下载训练镜像，不下载镜像test；官方测试数据未参与本次实验。

## 核心改动怎么解释

四个方向共用一个卷积编码器。每个方向产生1,600维特征，将对应特征取最大值，再进入同一个分类头。分类交叉熵直接作用于聚合结果，整个网络端到端训练。

对照模型也训练全部四个方向，并在推理时平均其预测，避免仅通过多看几张变换图像获得优势。普通预测平均和本模块都具有四方向不变性；本课题验证的是**训练时在特征层聚合的效果**，不能把“不变性”本身当成超过该对照的证据。

新增模块不增加参数，但需要四方向编码。这里没有声称单方向CNN能零成本获得旋转鲁棒性，也没有声称首次提出TI-pooling。
