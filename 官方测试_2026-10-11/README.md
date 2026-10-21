# 官方测试评价补充材料

先阅读 [官方测试结果报告](官方测试结果报告.md)。主结果为 10,000 张官方测试原图、3 个配对种子，开发集择优的简单融合对照 72.96%，四方向特征最大聚合 74.48%，平均提升 1.52 个百分点。

本目录补充在原项目归档外层，原 `rotation_cifar/` 快照及其中的训练代码、权重保持不变。此前整体压缩包保存的是官方测试前的归档；本轮补充材料单独保存。

## 文件用途

| 文件或目录 | 用途 |
|---|---|
| `官方测试结果报告.md` | 主结果、类别变化、与此前确认结果的比较及边界 |
| `results.json` | 全部种子、指标、配对区间和运行环境 |
| `predictions/` | 六个模型的逐图概率、标签、原图顺序 ID 及评价 JSON |
| `data/test.parquet` | 固定版本镜像中的官方测试部分，已核验 SHA256 |
| `data/official_test.npz` | 10,000 张测试图像、标签和顺序 ID |
| `data_manifest.json` | 数据来源、哈希、类别数量和重复检查 |
| `evaluation_lock.json` | 测试前固定的六个权重、来源文件哈希、融合选择和评价口径 |
| `evaluation_code_lock.json` | 正式推理前固定的评价脚本与协议哈希 |
| `evaluate_official_test.py` | 固定权重评价、旋转一致性检查、多种子汇总和配对区间 |
| `prepare_official_test.py` | 解码固定测试镜像，与原项目数据做完全重复检查 |
| `结果复核.json` | 下载和指标独立复核结果、修正及新增错误数 |
| `服务器结果SHA256.json` | 从服务器下载的 24 个文件的校验和 |
| `运行日志.txt` | 本轮实际运行输出 |
| `来源记录/` | 官网下载超时及切换镜像的记录；其中初版协议只作来源过程记录 |

NPZ 的 `probabilities` 形状为 `[4, 10000, 10]`。基线文件中的第一维是四个单方向预测，按锁定方式融合；模块文件中的第一维是对四种旋转输入实际运行完整模块所得的最终预测，已经包含内部四方向特征聚合。主统计使用原图为单位。

## 复查方法

阅读结果和从预测复算指标只需 NumPy。模型推理需要原项目的六份权重、固定来源文件及已有确认预测，以及兼容的 NVIDIA CUDA 环境；本轮使用 Python 3.10.19、PyTorch 2.5.1+cu121、NumPy 1.26.4。数据预处理额外需要 Pillow、PyArrow；本轮预处理使用本地临时依赖，没有修改服务器共享环境。

服务器执行位置为 `/home/hehaotian/ml-course-newtopic-20261011/rotation_cifar/official_test_2026-10-11/`。脚本以 `--project` 指定原 `rotation_cifar` 目录，运行时检查来源与权重哈希，并用此前确认结果检查指标与融合实现。

评价脚本禁止覆盖已存在的 `predictions/`。若要重新核对固定模型，可在本目录下建立工作副本，复制脚本、冻结清单、协议与数据，保留现有原始结果：

```bash
mkdir official_test_recheck
cp evaluate_official_test.py prepare_official_test.py evaluation_lock.json evaluation_code_lock.json 评价协议.md data_manifest.json official_test_recheck/
cp -R data official_test_recheck/
CUDA_VISIBLE_DEVICES=0 python official_test_recheck/evaluate_official_test.py --project ../rotation_cifar
```

以上命令从本目录运行，GPU 编号按空闲设备调整；单张 24 GB 4090 足够。脚本中的用时包含额外的旋转一致性核验，不作为公平延迟对比。再次评价仅用于复查既有结果，不应根据官方测试分数重新挑选方法。

独立补充压缩包需与之前的完整项目归档配合使用：将整个 `官方测试_2026-10-11` 目录解压到 `rotation_cifar` 的同级位置，即可对应这里的相对路径。权重不重复打包。
