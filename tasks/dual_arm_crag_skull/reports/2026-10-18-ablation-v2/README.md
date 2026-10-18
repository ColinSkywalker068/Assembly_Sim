# 头骨采样/联合损失消融与 PhysX 稳定性（2026-10-18）

本轮实际运行了 GPU 训练、9 标本验证、72 候选几何筛选及两次 Isaac Sim 稳定性对照。最佳平均中心误差 38.05 mm，但 72/72 候选均被几何检查拒绝；没有执行自主抓取或装配。数字是实验结果，不是可执行精度承诺。

## 参考接触图诊断

在完整头骨最大尺寸统一为 200 mm 的实验尺度下，对隔离碎片采样 4096 点，并查询到其余三角网格的距离：

|对象 / 隔离片|采样最小间隙|3 mm 内相反法向接触比例|结论|
|---|---:|---:|---|
|AMNH_103402_Skull / 4|2.334 mm|1.367%|更密采样可勉强连接宽松接触图，但仍有间隙|
|AMNH_196486_Cranium / 4|2.805 mm|0.439%|未达到 1% 接触比例|
|AMNH_76378_Skull / 7|3.966 mm|0%|超过接触距离|

这是采样距离，不能证明精确网格最小距离或断定缺片来源。未放宽筛选阈值。前两个对象从本轮训练排除；第三个属于原测试集，未用于模型训练/评估。详见 `contact-disconnect-v2.json`。

## 数据与控制变量

- 原 skull 源保留 7 个训练模型（6 标本）；验证由 2 扩至 9 个独立标本。新 7 件来自 CRAG MorphoSource v1/v2，仅用于验证；另 1 件非闭合网格被排除。
- AMNH/MCZ/RMCA/USNM 等馆藏编号分组，NMNH 与 USNM 视为别名；跨源去重，训练/验证标本不相交。旧 Hoolock 保持历史诊断集身份。原测试集未用于此次模型选择。
- 两种采样均通过网格/伪标签 QA，入选对象全部提前通过 512 点/片的参考装配筛选。没有依据预测效果删除验证对象。没有人工断裂面标签，分类 F1 衡量的是几何伪标签。
- 总预算固定 5000 点，最小分配 20→200；因此增加小片点数也减少大片点数。沿用官方归一化、旋转、法向和坐标恢复流程。
- 三个新实验共享同一 100-step encoder warmup 权重，随后解冻 encoder+denoiser，150 次迭代，lr=1e-5，训练种子 20261019，batch=1。AMP 实际更新次数为 145/146/145；预算相同不代表有效更新数完全一致。
- 位姿目标沿用官方 flow 向量 MSE；辅助项为有置信掩码的 Dice：`L = L_flow + 0.2 * L_Dice`。通过同一次 encoder forward 的特征计算辅助损失。它不是新增最终 SE(3) 测地损失。
- 固定最终权重，无验证集挑 checkpoint；noise=41/42，20 步×6 轮。所有组以同一 min20 标准点集计算中心/RMSE，分类指标也使用同一标准输入；固定锚片不计误差。
- shared warmup SHA256: `6e5efe94d45d5f5415d79162a912f7cb324e3bae58b6755a66d32893e9233e50`。各组 protocol、训练参数指纹、数据哈希见同目录 JSON。

## 结果

|方案|中心误差 mm ↓|旋转误差 ° ↓|点 RMSE mm ↓|断裂 F1 ↑|几何合格|
|---|---:|---:|---:|---:|---:|
|上轮 joint（历史比较）|43.22|81.07|47.79|0.565|0/18|
|新 min20，flow|39.13|78.47|43.05|0.573|0/18|
|新 min200，flow|38.05|78.25|42.70|0.543|0/18|
|新 min200，flow+0.2 Dice|38.70|76.24|43.19|0.573|0/18|

提高小片预算使平均中心误差再降 1.08 mm，但原两件验证上从 39.73 升到 45.04 mm，新七件从 38.95 降到 36.05 mm，效果并不一致。辅助损失恢复了分类 F1、改善旋转均值，但没有优于 min200 flow 的位置均值。历史组的数据/训练预算不同，仅三个新实验是匹配对照。这是单训练种子的短试验，不足以证明稳定泛化。

72 个候选全部检查所有碎片对，512 样本/片，双向接触、相反法向、薄壳包含/深穿透、接触图连通性；未使用 GT 误差选择部署候选。它仍是采样几何筛选，不是连续碰撞的精确证明。完整逐对数值见各组 `*-screening.json`。

## S2 工具与仿真

生产 JSONL 工具新增 `--minimum-points-per-part 200` 显式实验选项，默认保持官方 20。实际严格加载 multi200 checkpoint，通过历史 Hoolock 注册网格运行一次 seed42 调用；返回记录 min200、权重哈希、几何拒绝和 `motion_authorized: false`，见 `tool-integration.json`。该额外诊断使用生产默认 256 样本/片，不计入上面的 72 候选。

multi200 SHA256: `f5c3bd1564a696a252b754665f68f818a5b3c16c06477fa30b6dca048f53ec7e`。服务器权重在 `/local_data/jz6676/jing_assembly_run/skull-ablation-v2-run/{flow20,flow200,multi200}/joint.ckpt`；未提交大文件或提升默认模型。

Isaac Sim 5.1 / PhysX 中对原八片 Hoolock 体素诊断资产实测两次：每次 720 次 render world.step、24 秒、2880 个 120 Hz 物理子步，保存同步刚体位姿/速度以及三相机 RGB-D。不能把 world.step 次数称为物理子步数。

默认 solver 6/8 片稳定；仅将碎片求解器设置为 position16/velocity4 后 7/8 稳定，片1仍有约 19.5°/s 角速度。最终一秒窗口要求平移变化≤1 mm、旋转变化≤2°、线速度≤5 mm/s、角速度≤5°/s。窗口逐采样检查，不仅比较首尾。两次几何/质量/摩擦保持一致；未执行抓取、释放或装配。`smoke.json` 记录完整条件与实际时间。

机器人仍使用 Robotiq 代理；YAM、打印拓扑和机器人 base/flange/tool0 质量惯量警告尚未解决，不能称为已校准实机动力学。诊断体素资产尚未通过制造可行性审查。

## 复现及下一步

运行入口：`experiments.contact_disconnect`、`experiments.skull_dataset`、`experiments.skull_ablation_v2` 的 build/run/screen、`experiments.run_ablation_v2`、`experiments.visualize_ablation_v2`（均在 `tasks.dual_arm_crag_skull` 包下）。cohort 文件冻结后不可覆盖；完整源路径和样本标识见 `skull-ablation-v2-data/cohort.json`。

```bash
python -m tasks.dual_arm_crag_skull.experiments.visualize_ablation_v2 \
 --dataset ../skull-ablation-v2-data --run ../skull-ablation-v2-run \
 --output ../skull-ablation-v2-visuals --physics ../skull-settling-v2-solver \
 --physics-before ../skull-settling-v2-final
python -m pytest tasks/dual_arm_crag_skull/tests core/tests -q
```

本轮 CPU 回归 62 项通过，GPU 训练/推理和两次实际 Isaac smoke 均完成。图像保存在本地 `artifacts/skull-ablation-v2-2026-10-18/`（仓库忽略生成图），服务器为 `skull-ablation-v2-visuals/`。绘图按固定样本名取示例，没有挑最好看的改善。

下一步应做多训练种子/更长训练预算、按片尺寸分组评估以及显式接触/位姿约束，而非继续放宽阈值；修复片1碰撞代理与质量惯量问题，再推进物理夹持。只有候选通过几何门限，才能进入双臂 IK、路径和释放检查。
