# GARF 误差审计（2026-10-14）

结论：当前真实头骨 GARF 结果不能驱动自动装配。已复现官方推理，未发现坐标回代错误；更换 scheduler、法向符号、attention backend 或增加点数均未解决。根因尚未确认，不能把“领域差异”当作已证明的结论。所有毫米指标按整件最长边 200 mm 换算，并非实物尺寸测量。

## 已修复的实验随机源问题

官方 PointTransformer 的 `serialization` / pooling 在 eval 模式也调用 `torch.randperm`。旧适配器以候选 seed 设置全局 Torch RNG，导致初始位姿和编码器点序分组同时变化；“只改变 noise”这个说法不成立。当前已把位姿噪声独立为 Generator，并在隔离的 RNG 上固定编码器 seed=42。实测固定 seed 后重复编码仍有不同指纹（原因尚未定位，可能包含 GPU 算子非确定性），因此最终实现缓存最近一份几何的编码特征，按点坐标、法向和每片点数的 SHA256 失效。不同候选实际复用同一 latent，结果同时记录编码特征 SHA256，各候选 metadata 使用独立快照。

`multiseed_cached_encoder.json` 的4个候选指纹完全一致，cache hit 为 false/true/true/true。位置误差为60.11、51.95、52.94、57.01 mm；旋转误差112.73、114.86、120.79、122.08°。这验证了噪声隔离，尚未达到精度要求。下文的几何筛选报告针对旧4候选；新4候选未完成完整几何筛选，不能冒称同一结果。

修复后的官方流程（固定点云采样和编码器）仍失败：exponential 为 **91.47 mm / 96.61°**；linear + one-step 为 **82.63 mm / 100.53°**。这修复了实验可比性，但不是精度问题的完整解释。

最终缓存特征后的两片合并消融（仅统计剩余六片，成员由GT构造）也未达标：

| seed | 位置误差 | 旋转误差 |
|---|---:|---:|
| 41 | 50.58 mm | 144.50° |
| 42 | 55.25 mm | 125.32° |
| 43 | 45.10 mm | 135.07° |
| 44 | 84.29 mm | 93.41° |

最终官方 scheduler 对照共享同一份 latent，结果为 exponential：85.31 mm / 95.81°；linear：81.91 mm / 100.56°。两种设置均失败。

## 实验与结果（以下为固定编码器之前的历史结果）

| 条件 | 非锚点平均位置误差 | 平均旋转误差 |
|---|---:|---:|
| 官方 MeshInferenceDataset，固定采样，5000 点，app exponential | 93.51 mm | 95.22° |
| 同输入，linear + one-step init | 98.85 mm | 92.77° |
| 10000 点、小片最低100点，exponential | 96.90 mm | 110.30° |
| 同密集输入，linear + one-step init | 106.90 mm | 110.50° |
| 旧适配器，noise 41–44 | 46.6–58.9 mm | 101.3–124.4° |
| 理想两片合并，noise 41–44，仅剩余六片 | 35.78–81.21 mm | 102.17–151.66° |

最后两行不是同输入规模的受控比较，不能宣称合并提高成功率。合并使用 GT 构造实验条件，不是机器人完成了两片安装。内部接触面仍保留；是否去除、重采样、需要专门训练均待验证。

官方瓶子对照（12 片）：exponential 模式下 11 个非锚点中有 8 个恢复到 0.58–2.80 mm、0.09–0.25°；另 3 个仍严重失败。这支持当前权重/环境能输出正确配准，但不代表官方样例全部成功。

初轮 4 组 scheduler/法向符号消融误差 64.45–72.81 mm、117.47–119.48°；Flash 两组为 70.27–75.46 mm、122.27–123.86°。初轮和 Flash 各自使用未显式固定的 trimesh uniform 采样，不能用两轮差值量化 attention 效果。新脚本显式固定采样 RNG；最终主对照以 `skull_shared_features.json` 为准（两种scheduler复用同一特征）。

## 坐标与法向核对

官方先把整批 mesh 除以 `s = max(1, max_i max(mesh_i.extents))`，然后对每片采样、中心化、随机旋转、按最大绝对坐标归一化：

```
x = v / s
c_i = mean(x_i)
u_i = Q_i (x_i - c_i)
a_i = max(abs(u_i))
p_i = u_i / a_i
n_i_local = Q_i n_i
```

网络预测的 R、t 作用在恢复片尺度后的点：`x_hat = R_i (a_i p_i) + t_i`；mesh 单位恢复为 `v_hat = s * x_hat`。四元数为 wxyz，SciPy 输入转换为 xyzw。已知 GT 应满足 `R_i=Q_i.T, t_i=c_i`。固定采样头骨的点坐标回代最大误差 **1.0527e-8 m**，支持这些约定正确。

点云适配器的局部模型坐标变换为 `T.translation = common_scale*t - R*center`，不是仅乘尺度。法向只旋转、不平移、不按长度缩放；注册器统一法向长度。8 个原始 mesh 都闭合、绕序一致、带正体积；不能据此保证所有扫描局部面方向都正确。统一翻转法向没有恢复精度。

官方 mesh 输入与现有点云适配器仍有明确差异：官方按面积分配固定总点数5000、最少20点；适配器按输入点数分配512×片数、最少32点，并以当前局部点云轴向范围确定共同尺度。后者受局部旋转影响。下一步应显式保存模型尺度与面积采样预算，复用官方预处理；不能把本次审计描述为生产适配器已完全等价。官方路径自身已失败，因此这些差异不是已证实的唯一根因。

## 多候选筛选

`screening_surface_128.json` 是每片128个表面样本的首轮筛选：

| seed | 最差碎片对穿透比例 | 接触图组件数 | 结果 |
|---|---:|---:|---|
| 44 | 8.59% | 2 | 拒绝 |
| 41 | 14.06% | 2 | 拒绝 |
| 42 | 24.22% | 1 | 拒绝 |
| 43 | 74.22% | 1 | 拒绝 |
| GT（仅校准检查） | 0 | 1 | 几何筛选通过 |

位置误差未用于筛选。穿透阈值为超过1 mm的样本比例>2%；接触距离3 mm、双向接触样本比例>1%建边。这是未校准的拒绝启发式，不能作为成功概率或精确碰撞证明。

回归测试发现纯表面检查会漏掉完全重合的实体；当前实现额外检查内移探针，要求探针确实位于两物体内部。首轮报告保留原始表面版本，不冒充新实现的运行结果。新实现复用未变换 mesh 的空间索引，并在查询时反变换点。

`screening_probes_128.json` 已完成同一批历史候选的新版复测：seed41/42/43/44的最大穿透采样比例为17.19%/25.00%/78.91%/8.59%，全部拒绝；GT仍为0穿透、1个接触组件。该报告不含新缓存特征候选。

S2 JSONL 可通过 `--mesh-registry` 添加筛选；所有候选被拒绝时返回 `geometry_status=all_candidates_rejected`，始终 `motion_authorized=false`。即使几何通过，仍需 IK、路径、抓取、观测一致性验证。只有点云、没有闭合模型时，不可声称已经验证实体穿透。

## 可复现运行

在官方 GARF CUDA 环境、项目根目录执行（权重不在 Git）：

```sh
python -m tasks.dual_arm_crag_skull.experiments.official_audit \
  --repository /path/to/GARF --checkpoint /path/to/GARF.ckpt \
  --source /path/to/view_gt.glb --output /path/to/audit
python -m tasks.dual_arm_crag_skull.experiments.cluster_ablation \
  --repository /path/to/GARF --checkpoint /path/to/GARF.ckpt \
  --geometry /path/to/points-weighted --output /path/to/cluster.json
python -m tasks.dual_arm_crag_skull.tools.candidate_validation \
  --mesh-registry /path/to/mesh_registry.json --candidates /path/to/candidates.json \
  --output /path/to/screened.json --samples 128
```

`export_points` 现同时导出与 NPZ 严格同局部坐标、米单位的闭合 PLY 及 `mesh_registry.json`。mesh 不作为标签或 GT 传入打分函数。

## 自动装配仍未通过的环节

已有三相机 RGB-D/物理场景 smoke，不等于物理抓放或完整装配。本轮没有得到合格 GARF 候选，没有运行错误目标的自动机械臂装配。还需修复制造体素拓扑、验证夹持/路径和释放后稳定性，再接 S1 数据采集。当前 YAM 仍使用明确标注的 Robotiq proxy；原 FANUC 资产还有质量/惯量警告。

## 验证范围

本轮 `tasks/dual_arm_crag_skull/tests` 与 `core/tests` 共51项通过。扩大到旧任务全部测试时，两个渲染模块在收集阶段缺少 matplotlib；未据此宣称全仓测试通过。GPU验证包含严格权重加载、官方/适配器推理、共享特征指纹与合并消融。没有进行硬件运动，也没有取得真实头骨物理自动装配成功。
