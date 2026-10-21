# 头骨领域适配：第一轮实测

已执行真实 GPU 微调、独立标本验证和生产入口迁移推理。位置误差改善，但旋转和实体穿透仍不合格；没有运行候选驱动的机器人装配，也没有提升默认生产权重。

## 数据、标签与法向

数据来自 maxwell 上 CRAG release 的 `final_data/skull.hdf5`，共 391 个对象。筛出 36 个非下颌、至少六片的头骨模型；按 `AMNH_<标本编号>` 分组，先按固定哈希顺序划分，再进行质量剔除。同一标本的 Cranium/Skull 不跨集合。

| 集合 | 剔除前模型数 | 网格/标签审计通过的模型数 | 通过的独立标本数 |
|---|---:|---:|---:|
| train | 25 | 9 | 8 |
| validation | 5 | 2 | 2 |
| test | 6 | 4 | 4 |

19 个模型至少一片不闭合；另 2 个模型在 5,000 点中只有少于 20 个高置信接触正例，未用于本轮。剔除原因和完整清单在 `dataset_manifest.json`。测试集仅做数据几何 QA，没有训练或模型推理评分。

**391 个对象均没有 `shared_faces`。** 官方 HDF5 loader 缺字段时会用 -1 填充，进一步变成全负断裂面标签。本实现将缺失视为未知，禁止把它当作已标注负例。这是微调数据管线问题，不能据此断言原始预训练权重曾用这些错误标签训练。

使用几何伪标签，而非人工语义真值：

- 正例：另一片最近表面距离 ≤ 全件最大尺寸的 0.0005，且法向点积 ≤ -0.8。
- 负例：到其他片最近表面距离 > 全件最大尺寸的 0.002。
- 中间区域以及邻近但法向不相反的点忽略，不参与 Dice 损失。
- 统一到实验装配最大尺寸 200 mm 后，两距离分别为 0.1 mm / 0.4 mm；该缩放不是原始标本的物理尺寸声明。
- 所有训练输入复用官方 MeshInferenceDataset 的 5,000 点面积分配、至少 20 点、中心化、随机旋转、max-abs 尺度及法向旋转。采样 seed=42。
- 通过样本的坐标/法向往返最大误差分别为 8.14e-8 / 8.28e-8。正例法向点积中位数范围为 -0.99918～-0.99825。
- 独立检查：正例到提供的完整头骨表面的距离中位数，是负例的 68～1371 倍；支持这些点来自内部断裂面。完整表面已简化，不能把此检查当逐点人工标注。

## 微调协议

固定 100 个迭代步骤/阶段，batch=1，AdamW，学习率 1e-5、weight decay=0.01、梯度裁剪 1；训练 seed=20261018。训练按模型洗牌，应用官方中心化/随机 SO(3) 旋转，法向和目标四元数同步变换。BN 使用固定运行统计，仍允许参数梯度。

1. **baseline**：用户给定原权重。
2. **encoder**：显式解冻官方默认冻结的 feature extractor，包括 encoder、分类头和 BN affine；使用带不确定区域掩码的 Dice。denoiser 保持冻结。
3. **joint**：接着 encoder checkpoint，联合更新 encoder 和 denoiser；使用官方 SE(3) flow-matching `vec_mse_loss`，没有额外分类损失。

AMP 溢出导致两阶段分别跳过 2 / 4 个更新，实际更新 98 / 96 次。参数指纹证实第一阶段 encoder 变、denoiser 不变；第二阶段两者均变。`*-training.json` 保留全部步骤，不以 loss 下降代替参数更新验证。

验证固定两个未见标本，每个 noise seeds 41、42；20 步 × 6 次官方应用；同一采样和 encoder 特征缓存。只报告固定最后一步 checkpoint，没有验证集选 checkpoint 或超参数搜索。每个物体先平均噪声结果，再等权平均物体。主误差排除已知锚片，使用非锚片采样中心误差和旋转测地角。输入不包含其余片的 GT 位姿；训练监督和离线评估可以读取 GT。

| 阶段 | 中心误差 mm ↓ | 旋转误差 ° ↓ | 点 RMSE mm ↓ | 断裂面 F1 ↑ |
|---|---:|---:|---:|---:|
| baseline | 64.52 | 110.76 | 68.03 | 0.179 |
| encoder | 53.02 | 112.04 | 55.37 | 0.533 |
| joint | 40.91 | 90.49 | 41.62 | 0.411 |

这是 **2 个验证标本、单次训练种子、顺序微调的小规模 pilot**，不能作为稳定泛化结论。一个验证模型以几乎完整的头骨为大锚片，其余是极小碎片；视觉上保留完整头骨不代表模型成功。图中明确标记固定锚片，分数排除它。joint 的分类下降提示后续应预注册比较分类辅助损失；当前结果不能证明所有误差都由断裂面分类导致，因为分类掩码不直接输入位姿 denoiser。

## 发现并修复几何筛选误报

两个参考装配最初被报告 66%～69% 最大穿透。独立实体绕数对每件 16 个疑似深穿透点均给出约 0，即空腔；另一套 float64 三角射线也判为外部。

根因是安装的 trimesh Embree 路径在输入坐标中使用 `1e-4 * (100 / mesh.scale)` 的多交点射线偏移。在米制小网格上会跳过薄壁。生产检查改为显式中心化、归一化到固定 100 单位，再关闭 Embree 内部缩放。修复后这些 32 个点与实体绕数判断完全一致。增加空腔/壁内/外部、跨 0.001/1/1000 尺度的回归测试，实际 Embree 环境通过。

同时修复大小片接触图：原先要求双方均超过 1% 面积，容易漏掉大头骨上的小接触区；现在要求任一方向超过 1% 采样接触，且相邻法向点积 < -0.5，保留双向分数供审计。这仍是采样检查，不是完整连续碰撞或运动可行性证明。

修复后，两个验证参考装配均为 0 检出深穿透、1 个接触分量；**全部 12 个模型候选仍然被拒绝**。修复前记录保存在 `baseline-screening.json`，已被 `corrected-screening.json` 替代，不能混用两版筛选分数。全体网格合格样本的额外参考审计另见 `reference_qa.json`；该审计在第一轮微调后补做，不伪称训练前已全部通过。

全体 15 个样本均未检出深穿透，其中 12 个接触图连通。`AMNH_103402_Skull`、`AMNH_196486_Cranium`（训练）和 `AMNH_76378_Skull`（保留测试）的参考接触图有两个分量，尚需区分采样漏检、间隙或缺片。本轮训练包含前两者，应视为诊断 pilot；新增 `skull_finetune --reference-qa /path/reference_qa.json` 可在下一轮先排除它们，得到 7 个训练模型/6 个标本、原 2 个验证标本。未重新训练或把新门槛追溯写成旧实验条件。

## 原八片 Hoolock：生产入口迁移检查

把保存的 `joint.ckpt` 严格加载进 OfficialGarfBackend，通过真实生产候选工具运行 seeds 41～44。此对象未加入训练，但此前已反复诊断，属于**历史迁移检查**，不是新的盲测集。

| 权重 | 原生产指标：平均局部原点位置误差 mm | 平均旋转误差 ° |
|---|---:|---:|
| baseline | 97.13 | 126.89 |
| joint | 38.31 | 106.48 |

这组位置指标保持原生产 `evaluate.py` 定义，不与上表采样中心误差混为同一指标。四个 joint 候选全部未通过修复后的几何筛选。历史 baseline 四候选也已重筛，记录在 `historical-hoolock-baseline-rescreened.json`。S2 可以加载新权重生成研究候选，但不得将它自动视作可执行计划。

## 复现与文件

上游 GARF 固定 `2d489c0b7b58de8af774aa1657c86c638e4744c2`；数据、基础权重和两份新权重指纹见 `summary.json`、`protocol.json`。权重、原始网格、点云和生成图片不提交仓库。

远端目录：`maxwell:/local_data/jz6676/jing_assembly_run/`。

- `skull-adaptation-v1/`：全部对象审计、划分和官方采样 NPZ。
- `skull-finetune-v1/{encoder,joint}.ckpt`：实际模型 state_dict，可由现有 backend 严格加载。
- `skull-finetune-v1/`：训练曲线、全部验证预测、几何筛选。
- `skull-finetune-hoolock-v1/`：原八片完整生产推理输出。
- `skull-adaptation-visuals/`：标签图、逐片图、两个验证物体的装配对照、指标图、损失图、Hoolock 迁移图。

从 Assembly_Sim 根目录运行以下模块（完整 CLI 见各自 `--help`）：

```bash
python -m tasks.dual_arm_crag_skull.experiments.skull_dataset --source /path/skull.hdf5 --repository /path/GARF --output /path/dataset
python -m tasks.dual_arm_crag_skull.experiments.skull_finetune --repository /path/GARF --checkpoint /path/GARF.ckpt --dataset /path/dataset --output /path/run --steps 100
python -m tasks.dual_arm_crag_skull.experiments.skull_screen --source /path/skull.hdf5 --run /path/run --output /path/run/screening.json
python -m tasks.dual_arm_crag_skull.experiments.skull_visualize --dataset /path/dataset --run /path/run --output /path/figures
```

CPU 回归 57 项通过；另外在实际 Embree 环境运行 4 项几何回归，全部通过。没有新物理装配成功记录。下一步应先处理未通过参考接触图的样本、扩充独立标本验证集，比较小片采样预算和分类/位姿联合损失，再依据合格几何候选推进 IK、路径和 PhysX 夹持验证。
