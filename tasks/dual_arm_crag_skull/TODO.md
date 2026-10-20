# Jing 实施 TODO

勾选表示已完成；其他项目是计划，不能视为已有能力。

## S2 与 ASPIRE 统一（当前）

- [x] 默认 S2 改为 ASPIRE 冻结 Python 程序；skills / direct 两模式共享 `s2/aspire.py` 的上游程序运行器。
- [x] skills 模式保留候选解析、技能/手臂组合校验与编排器，输出请求交给 S1/S0；不开放直接运动 API。
- [x] direct 模式复用 ASPIRE trial runner、种子守卫、S0 检查与评分；结果仍归 ASPIRE，记录 execution_mode。
- [x] 旧 Claude 工具规划器保留为显式选项；增加 skills 计划 CLI 和 direct trial CLI。
- [ ] 接通 ASPIRE coding-agent 的头骨任务生成—试跑—修正循环及成本/试验预算计量。
- [ ] 提供真实训练 checkpoint 的 S1 执行器与 Isaac 后端，验证两模式物理闭环。
- [ ] 增加程序运行的独立进程墙钟限制；上游 AST 检查不构成隔离或 watchdog。

## P0：本次脚手架

- [x] 建立本地 jing 分支与独立任务目录。
- [x] 固定 S2 / S1 / S0 职责与双 TCP 动作格式。
- [x] 增加配置、结构检查 CLI、README、接口文档和 TODO。
- [x] 核实 Claude Opus 5.5 的 API model ID（`claude-opus-5-5`，取自 Anthropic 模型表）；实现供应商适配器 `s2/claude.py`、strict 工具 + 契约校验、显式选择的离线 `ScriptedPlanner`。
- [ ] 真实 API 调用验证：需要你提供凭据并同意把观测（含关键帧）发给外部服务；目前只用假客户端测试过。
- [ ] 确认真实 FANUC 型号/控制器/外部控制选项；目前暂用两台 CRX-10iA/L。
- [ ] 确认 YAM 具体型号、夹爪资产、安装板、开口范围、供电通讯及 TCP。
  - 10-18 已定型号：I2RT `linear_4310` 线性夹爪（用户给的产品页 i2rt.com/products/yam-gripper-1，原文 "the 'Linear 4310' gripper in Github Readme"，行程 95 mm）。
  - 资产来源：官方仓库 github.com/i2rt-robotics/i2rt（MIT，v1.3.6 = `120c3c8`）的 `i2rt/robot_models/gripper/linear_4310/`（MJCF + STL，没有单独的 URDF；整臂 `arm/yam/v1/yam.urdf` 里带同一夹爪）。
  - 官方数字：开口每指 0–47.5 mm（URDF 写 46.95 mm，两份差 1.1 mm，要量实物）；TCP 在指尖端面中点，沿安装面 +Z 144.65 mm，开口变化时不动；本体 0.553 kg，每指 0.071 kg。
  - 还缺：CRX 法兰到夹爪的转接板（厚度、绕法兰的安装角度）、夹持力、实测开口与 TCP。
- [x] 记录上游 commit（df03da2）；复现 Arm(dict)/Arm(path)：`teleop.py:40` 传 dict 时 `TypeError`，现在 `Arm`/`Gripper` 接受路径或映射（`core/tests/test_kinematics_input.py`）。
- [x] 修复 small shulker `config.py` 的跨任务导入（`tasks.stack_sawtooth_shulker.conditions` 从未存在，函数就在本任务 `conditions.py`），该任务 5 项失败测试恢复。
- [ ] `core.robots.teleop`（`CartesianInput`、`KeyboardSubscription`）在全部历史中都不存在，`teleop.py`/`scripted.py` 无法导入，`test_scripted.py` 收集失败；需要原作者补回实现，不凭空复原。

验收：配置检查不会启动仿真或硬件；未知硬件参数显式留空。

## P1：同源碎片与制造

- [ ] 定位 CRAG 头骨原始 GLB/mesh、GT 与预测变换、数据使用条款。
- [ ] 实现 CRAG loader：保留节点变换与稳定 fragment ID，校验 N >= 6，优先八块。
- [ ] 实现统一尺度/网格与唯一体素归属，保存原始到制造坐标变换。
- [ ] 检查六邻域连接、薄壁、流形、内部面清除、体积和插入路径。
- [ ] 导出 print.3mf/STL、visual mesh、collision assets 与 geometry manifest。
- [ ] 打印间隙试样；测量尺寸、质量与摩擦，再确定整套参数。
- [ ] 若需支撑底座或连接件，将其纳入几何和任务说明。

验收：每片可打印、目标拼接无不合理重叠；仿真与打印尺寸、单位、几何版本可追溯。

## P2：双臂和三相机

- [ ] 抽象 Robotiq 专用路径/驱动，接入实选 YAM 模型。
  - [x] 10-18：官方 `linear_4310` 模型放进 `core/assets/robots/i2rt_linear_4310/`（sha256 钉住）；`core/robots/linear_4310.py` 读出网格、TCP、滑动轴、开口映射，与 MuJoCo 逐项核对；S0 的 `yam_model` 用暂定 15 mm 转接板，种子 101 贴桌检验 7/7。26 个变异全部被测试抓到。
  - [x] Isaac 场景里按夹爪描述组装 YAM（`sim/yam_usd.py`，只在头骨任务里使用，上游 Robotiq 组装没改）；`sim/check_yam.py` 在 Isaac 5.1 通过：开口误差 ≤0.0003 mm，PhysX 里的法兰、本体、指尖与 S0 模型差 <0.001 mm / 0.0001°，S0 的 TCP 与 PhysX 指垫给出的抓取点差 0.08–0.11 mm。
  - [ ] 夹爪驱动的真实值（刚度、阻尼、最大力；现在 20 N 是暂定值）；两指耦合（现在是两路独立驱动，下发同一目标）。
  - [ ] 转接板设计（厚度、安装角度、孔位）后把 `s0.flange_T_tcp` 写进 `jing.json`。
- [ ] 加载左右基座、法兰到工具和三相机标定。
- [ ] 创建物理 CRAG 场景，验证刚体质量、碰撞与稳定初始摆放。
- [ ] 三路图像同一仿真时刻采样，保存曝光时间与实际状态索引。
- [x] 离线 FK 与 Isaac probe 的 link/flange 位姿比对：最大 0.093 mm / 0.0089°（probe 记录到四位小数）。
- [x] 在 PhysX 中多构型比较 TCP 位姿；验证夹爪开口与运动范围（10-19 `sim/check_yam.py`：开口误差 ≤3e-7 m，法兰 / 本体 / 指尖与 S0 模型差 <0.001 mm，TCP 与 PhysX 指垫给出的抓取点差 0.11 mm，四组方块夹持与网格预测差 ≤0.1 mm）。

验收：单块能物理抓起、保持、释放；无物体 attach/teleport；三相机有效。

- [x] 10-19 通过（`sim/check_grasp.py`，`runs/check_grasp_20261019_002209`，判据跑之前写死）：种子 101，左臂用脚本经 S0 抓 `fragment_3b11f6f1187b8b2b`（72×56×48 mm，弦 56.4 mm）。
  - 夹紧宽度 56.40 mm（弦 56.43 mm）；上提 99.8 mm，保持 2 s 相对 TCP 滑移 0.0009 mm；平移 5 cm 误差 0.28 mm；放回后高度差 0.0003 mm、位置差 0.008 mm、倾斜 0.0001°、静止；单步最大位移 0.53 mm（无瞬移，后端从不 attach）；其他碎片与锚点 <0.001 mm；三相机有效。
  - 前提：S0 用测试值；YAM 转接板与夹持驱动是暂定值；位姿是仿真真值（`oracle_sim`）；脚本抓取不是 S1；只验了一片、一个种子。
  - 过程中修了三处物理问题：手臂驱动只给位置目标时有 0.1 s 一阶滞后（改为速度前馈）；PhysX 默认凸分解让指垫每侧鼓出 1.8 mm（改为 2 mm 薄片凸包）；选抓取只用网格顶点会把体素碎片量窄（改为密集表面采样）。
  - 10-19 独立审查（4 HIGH 全部核实）后重跑，三项检查都通过（`runs/isaac_checks_20261019_024300`）：
    - 被挡在指令外 5.98 mm；挤压到搬运结束相对 TCP 偏移 0.039 mm；放回 0.11 mm 并静止；没有关节连到碎片，碎片为动态刚体。
    - check_yam：10 个探针（含窗口里的 3 mm 杆）与网格预测差 ≤0.005 mm；704 个碰撞体烹制后全部有效，凸出 0，内缩 ≤0.083 mm。内缩的容差 0.1 mm 是看到这个数之后才定的，代码里已写明。
    - 过程中又修了两处：手指驱动从全开一步合拢实测 1.5 m/s，会撞穿方块，加了暂定的最大速度 0.1 m/s；小碰撞块的自动接触偏移太小，显式设为 2 mm。
- [ ] 腕部相机：按 Robotiq 设计的安装位姿在 YAM 上看不到夹爪和被夹碎片，要按 YAM（和 UMI 手持工具）重新定。
- [x] 抓取规划的手指模型（10-19，审查 H3/M1）。⛔原记录「平指垫 20 mm、往上外退 6–9 mm、0.48 mm 浅槽」是错的：手指只有最末 15 mm 实心，往上两根侧梁夹一个贯通窗口；35 mm 处侧梁只比指垫面退约 1.1 mm。
  - `sim/fingers.py` 从网格算内伸量图（0.5 mm 格），`choose_grasp` 改用它：弦长取合拢时最先接触的宽度，TCP 平移到两指同时接触，张开要清掉下降途中扫过的整根手指。
  - 碰撞体按片内连通性拆开窗口，每块 ≤64 顶点；`check_yam` 加了 3 mm 杆形探针和烹制形状读回。
- [ ] 手指根部（离指端 ≥57 mm）截面呈 L 形，凸包填角，74–80 mm 处向内多伸 48 mm。要做根部的凸分解，否则不能抓高于约 6 cm 的物体。
- [x] 保护性停机（10-19 用户决定，已实施并冻结）：`configs/jing.json` `s0.protective_stop` = 跟踪误差 > 10 mrad 连续 3 个控制周期，或关节速度 > 1.1 × 最高速度 → `ProtectiveStop`；Isaac 后端每周期检查（`baselines/backend.StopMonitor`）；DexAgent 执行器与 ASPIRE 适配器/运行器结束试验、不评分、记 `protective_stop`（失败）并留日志（`execution.json` / `protective_stop.json`）。阈值依据：Isaac 上正常抓放最大 0.32 mrad（DexAgent 橙色交付含放下）、ASPIRE 参考 0.18、自由 1.2 rad/s 斜坡 0.07；锚点卡死 279 mrad、4.9 rad/s，此规则在接触后 0.3 s 触发。测试 + 变异 14 处全部抓到（同环境基线先过）。Isaac 上的两条验证（正常不误停、卡死被停）见日志。
- [ ] CRX 资产里 10 个刚体的 MassAPI 质量和密度都是 0，PhysX 按碰撞体体积乘默认密度算质量。刚性位置驱动下对跟踪影响很小，但要做限力或力控时需要真实惯量。
- [ ] 种子 101 的筛选：96 mm 那片任何方向都超过 YAM 94 mm 开口，顶抓不了；有几组被拒是因为另一只手臂停在初始位挡路（臂间间隙），需要双臂协调（另一只手让开）。

## P3：S0 与专家示范

- [x] 实现统一 ActionChunk 编解码、时间有效性和单位检查（`contracts/actions.py`）。
- [x] DLS IK 基线（失败给出结构化原因，不返回裁剪解）+ 关节限位/速度/加速度拒绝 + 伺服采样点上的胶囊体两臂与桌面检查（`s0/`）；双臂速度级盒约束 QP 跟踪器（`s0/tracker.py`，无距离约束）。
- [ ] 连续碰撞检测、单臂自碰撞、基于网格的精确几何；QP 加局部线性化距离约束。
- [ ] S0 参数待定（`jing.json` 中为 null）：关节加速度上限（资产与公开资料均无）、YAM 开口范围与 `flange_T_tcp`、两臂间隙、桌面间隙、TCP 最低高度、观测年龄上限、anchor 位置/转角容差、接触阶段的指尖余量 / 区域半径 / 速度上限。
- [x] 独立代码审查（2026-10-18）的 6 条 HIGH 与 13 条 MEDIUM/LOW 全部修复并补测试：编排器执行前复核与规划时长上限、失败计数不被 reobserve 清零、合并簇条件、GARF 客户端失步检测与进程组超时、S0 制动轨迹、夹爪包围盒桌面检查、anchor 与观测时刻 TCP 比对、动作块重放拒绝、S0 观测年龄上限、内参半像素约定、fallback 回填规则。
- [ ] 制动轨迹本身未做碰撞检查；块与块交界处加速度跳变、jerk 不受约束（实机 J519 的要求未核实）。
- [x] 贴桌抓取的接触阶段（2026-10-18，先写测试再实现，`tests/test_s0_contact.py`）：
  - 动作块按技能声明 `ContactWindow`（只有抓 / 放两种），S1 运行时自动附上；retreat 不单独声明，沿用这只手上一次的抓 / 放窗口。
  - S0 只在常规桌面 / TCP 规则失败的采样点查声明，并且要求：
    - TCP 和手指盒低于底线的全部部分都在声明中心的半径内；
    - 手指盒和 TCP 高于接触余量，本体盒保持常规余量；
    - 相邻伺服区间不超过一个伺服周期，手指盒角点限速；
    - 同一请求内接触不能移动。

    细节见 INTERFACES.md「S1 -> S0」。
  - 独立审查三轮：
    - 第一轮 2 HIGH / 4 MEDIUM / 8 LOW。HIGH 是：retreat 能在任意位置下探、沿桌面爬行；速度规则缺测试。
    - 第二轮发现给 retreat 打的补丁仍可让它沿桌面慢慢下沉、横移，所以改为 retreat 不单独发窗口。
    - 第三轮没有 HIGH。两条 MEDIUM 已修：
      - retreat 原来继承"最近一次声明过"的窗口，而且永不清零，改为只记 S0 接受过的（`ContactMemory`）并按 episode 清零；
      - 另有 6 个变异体存活，已补测试。
    - 其余已修改并补测试：M1–M3、L1/L3/L4/L5/L7、区域按盒棱交点算、间隔改用相对容差、采样时间严格递增、伪造的声明只会得到制动。
    - M4（接触带外的接近速度）和 L2（扶持手）写进了文档。
    - 最终 50 个变异全部被测试抓到。`ContactMemory` 这最后一处改动只经过测试和变异检查，没有再送独立审查。
  - 真实头骨种子 101 的检验（`experiments/s0_contact_check.py`，结果 `reports/s0_contact_2026-10-18.json`；S0 用测试值，区域半径 0.12 m）：7 片移动碎片上方竖直、指尖离桌 3 mm 的位姿，不声明接触全部被拒；按 pick 声明接触后，7 片都有至少一只手被接受。
- [ ] 接触阶段的后续：
  - 三个参数的真实值；
  - ASPIRE 程序还不能声明接触（见 P6）；
  - 扶持中的非活动手不能声明接触；
  - 手臂停在接触带里、记忆里却没有它的窗口时（例如重启后），只能人工恢复；
  - 碎片不是 S0 的碰撞对象，邻近碎片没有保护；
  - 接触时没有力控，只靠位置和限速；接触带外的接近速度只受关节限制。
- [ ] 接入全局搬运路径规划接口，不将局部 IK 当作避障规划。
- [ ] 完成单臂交替装配区占用；后续增加双臂支撑/插合模式。
- [ ] 多碎片技能、顺序与可达性专家；阶段检查及失败原因。
- [ ] 分别记录请求、接受命令和执行后实际状态；异常退出保存部分数据。
- [ ] 建立逐块误差、释放后稳定性、最终完整性评估，避免只检查轨迹结束。

验收：至少一条 >=6 块物理装配示范、成功与失败可复查、无时间对齐错误。

## P4：Diffusion Policy S1

- [x] 三相机编码与 conditional U-Net1D：原生实现 Chi et al. 结构（ResNet-18 + GroupNorm、FiLM、DDPM/DDIM、EMA），不依赖上游 DP 仓库；训练环境版本钉在 `s1/requirements-train.txt`。
- [x] episode 格式（schema 2：`meta.json` + npz，privileged 单独存放且训练从不读取；记录 `initial_condition_id`、相机接收时间与时钟域、技能段的碎片观测位姿）、10 Hz 重采样（SLERP，拒绝外推）、严格因果的状态取样、图像到达后才可用且帧龄从曝光算、缺失遮罩、20D SE(3) 动作编码、监督走到技能段末并保持最后一步、按初始条件划分、只用训练集拟合的归一化（`s1/episode.py`、`s1/dataset.py`）。
- [x] 数据卫生：默认只用成功示范（`include_failures` 显式开启）；oracle 目标或仿真真值碎片位姿报错，除非 `allow_oracle`；跨 > 0.1 s 位姿空洞或状态过旧的窗口丢弃并计数。
- [x] S1 独立审查（3 HIGH / 8 MEDIUM / 4 LOW）全部修复，每条都有测试，22 个变异全部被抓到。checkpoint 记录条件编码布局，不一致拒绝加载。
- [x] 合成数据 GPU 冒烟管线与运行前写定的判据（`s1/smoke.py` v3）。审查前的 torch 作业 20219996 / 20221507 已作废。v4（审查修复后）未过 C3/C10；v5 预登记（eta=1、16000 步、新数据）策略质量各项通过，延迟两项在被占满的共享 GPU 上未过（`reports/s1_smoke/README.md`）。只说明管线和方法可用，不代表头骨装配能力。
- [x] 在不争用的 L40S 上判定 v5 代码包：torch 20232734（预登记）十项全过，DDIM-16 52 ms、部署路径 54 ms，训练损失与 maxwell 逐值相同；v4 包 20226851 同样 C3/C10 不过、数值相同。已并入 `reports/s1_smoke_2026-10-18.json`。
- [ ] 决定部署默认的 `sampler_eta` 与训练步数。合成数据支持 eta=1、16000 步；`PolicyConfig.sampler_eta` 仍默认 0，这样没有该字段的旧 checkpoint 加载后仍按 eta=0 采样。头骨任务的取值要等真实示范数据上的同类检查再定。
- [ ] 用真实分辨率三视角、裁剪与预训练编码器测试；流式数据加载（目前整个 split 预载到 GPU）；episode 补关节诊断与动作下发时间。
- [x] S1 以碎片观测位姿（`fragment_pose`，相对第一只活动手的 anchor）和活动臂掩码为条件；checkpoint 记录训练过的 (skill, arms) 组合，S1 拒绝其他组合，S2 拿到这些组合后在规划与执行前同样拒绝（`contracts.skills.trained_combination_set`）。
- [ ] 仿真专家先做少量闭环基线，再按实测成功率扩大数据量；不预设 50 条足够。
- [ ] 随机摆放、可观测目标、材质/光照/相机扰动；校准后再做动力学随机化。
- [ ] 分阶段训练 pick/place，再训练 align/insert；加入经验证的恢复示范。
- [ ] rollout 测试：未见初始条件、相机扰动、碰撞/抓取/装配失败分类。

验收：checkpoint 可复现；报告 closed-loop 成功率、置信区间、数据分布与全部失败。

## P5：UMI / 实机 / S2 集成

- [ ] 双手采集器和机器人端匹配指尖与相机/TCP 几何；采集同步第三视角。
- [ ] UMI SLAM、左右坐标对齐、开口标定、质量筛选和时钟同步。
- [ ] 导入统一 episode，保留 source/缺失通道/不确定性，仿真验证动作可达性。
- [ ] FANUC 原生伺服 + 官方 ROS 2 SJTC 运动适配；核实 J519/R912 或 S636、固件与双控制器同步。YAM 使用独立驱动，实测延迟、限制和跟踪误差。
- [x] S2 在完成/失败时输出已训练技能的 skill request；请求经契约校验，目标位姿由编排器从记录解析，不直连关节驱动（假模型客户端测试）。
- [ ] 比较固定顺序与 S2 重规划；比较 sim-only 与 sim+UMI；物体 GT 仅用于显式 oracle 消融。
- [ ] 若插合需柔顺，基于实际传感器与控制能力实现并验证导纳控制。

验收：先无接触回放，再单片抓放，再多片装配；每阶段有日志与独立结果评估。

## P6：基线（DexAgent / ASPIRE）

已完成（2026-10-18，全部离线，没有基线成绩；见 `baselines/README.md`）：

- [x] 公共层：`AssemblyTask`、种子初始条件与三段种子、`score_assembly`/`RunRecord`/Wilson 统计、留出种子去重（`refuse_repeats`）、`AssemblyBackend` 接口（相机帧带采样时间）、测试用 `KinematicBackend`、S0 `check_joint_path`、命令行；真实头骨任务记录已导出到 `runs/baseline_tasks/`。
- [x] ASPIRE（`fa00f0e`）：补丁（`fanuc_crx`、`crx_dls`）与 checkout 校验、CRX 适配器（逐命令 S0 检查、周期间几何、末端保持周期、被拒夹爪命令抛出）、IK、冻结程序 runner（种子守卫、留出 trial 必须记录且只跑一次）、CLI、参考程序；运动学替身上端到端测试通过。未调用模型，学习循环未运行。
- [x] 公共层独立审查（4 HIGH / 6 MEDIUM，全部修复并补测试；审查留下的存活变异和新行为共 54 个变异全部被测试抓到）：
  - 留出种子在试验前原子占用，崩溃或中断也记为 `error`，换记录文件也不能重跑；
  - 协议固定为 `jing.json`，`verify_task` 从 layout 重建核对任务；
  - 评分前统一"放手并静置 `settle_s`"，非有限位姿判失败，GARF 候选目标的任务拒绝评分；
  - `check_joint_path` 对非有限状态或时钟直接拒绝，改为匀加速度口径（第一段 2 倍、非均匀间隔取平均），并加末端制动检查（ASPIRE 的 `action_bounds` 随之改为 a_max·dt/2）；
  - `summarize` 拒绝测试替身记录、重复的留出种子、矛盾的 success、非标准种子段，以及同组混用不同判据；
  - 任务 id 带 layout 哈希，layout 增加几何检查。
- [x] DexAgent（`e29bdad`）：原生 YAM + Sharpa 机器人，只跑 Stage 3；从任务构造 Stage 1/2 输入和替身视频，全部通过它自己的门禁；模型中转（Opus 5.5、effort high、保留 thinking、`drop_block` 绑定、每次调用前检查美元上限、密钥不进 DexAgent 进程）；交付从新克隆复核后用我们的评分；mock LLM 测试通过。没有调用模型。

待完成：

- [x] **Isaac Sim `AssemblyBackend`**（10-18，`sim/backend.py`，在 maxwell 的 Isaac 5.1 上由 `sim/check_backend.py` 验证通过）：物理 240 Hz、关节与宽度目标 60 Hz（每周期 4 步，关节目标线性插值）、相机 10 Hz 按需渲染并缓存（暂停物理，渲染完用 `World.play()` 恢复）；观测位姿是标为 `oracle_sim` 的仿真真值；特权位姿只给评分；锚点是运动学刚体（夹具）；`reset` 先让碎片落定（最近 0.5 s 漂移 <0.1 mm / 0.05°）再从 0 计时，没落定就拒绝开始；YAM 两指是两路独立驱动。
  - 检验结果：两个种子都在 1.0 s 落定，`fragment_36d5…` 倒 18.44°（与 MuJoCo 的 18° 一致）；锚点 2.6e-9 m；时钟逐步精确；保持 1.4e-4 rad；夹爪 30 mm 误差 1e-8 m；跟踪 8e-4 rad；三相机外参与当前位姿差 <0.001 mm，渲染不推进物理，缓存生效，渲染后物理照常推进。
  - 独立审查（1 CRITICAL / 3 MEDIUM / 若干 LOW）已处理：CRITICAL 是第一次渲染后物理冻结、时钟照走（`timeline.play()` 没提交，`World.step(render=False)` 在未播放时静默跳过物理），检验脚本因为最后才渲染没发现；另修了目标数组与调用方共用、没落定的起点不拒绝、两指碰撞未过滤、`check_yam` 的 TCP 检查自证循环等。变异检查 74 个全部抓到（只有一个因夹爪对称而等价）。
  - 还没接：ASPIRE 适配器跑在 envs/aspire，Isaac 跑在自己的解释器里，需要进程边界或在 Isaac 环境装 `cap_harness`；适配器的宽度范围与 S0 限值要换成 YAM 的（93.96 mm）；`evaluate_frozen_program` 每个种子新建后端，Isaac 后端每进程只能有一个。
  - 还没做：夹住已知宽度方块的检验（验证指垫凸分解接触几何）；两指耦合（`PhysxMimicJointAPI`，MJCF 是 1:1 等式约束）；落定后的起点写进 `RunRecord`（`backend.start.to_record()` 已有）；`fragment_36d5…` 倒下后重新检查足迹，或修改它的摆放位姿（改 layout 等于改任务，要在任何方法看留出种子之前定）。
- [ ] `jing.json` 中 S0 的真实值（加速度上限、夹爪宽度范围、间隙、TCP 最低高度、接触阶段三个参数）与真实夹爪几何。
- [x] **ASPIRE 的接触声明（公平性）**（10-19 完成，`baselines/aspire/README.md`）：
  - 程序调用 `fanuc_crx.declare_contact(kind, fragment_id, arm)` 和 `clear_contact`；补丁注册了这两个工具，任务上下文写明规则（`metadata.contact_rule`）。
  - 程序只能报碎片 ID：抓取窗口按该碎片的观测位姿算，放置窗口按它的目标位姿算，程序给不了坐标。
  - 手臂在接触带里时不能声明或清除窗口（先抬出），所以窗口留在声明的地方，与 S1 的 retreat 规则一致。
- [x] **锚点稳定性（任务定义问题，所有方法一致处理）**：锚点碎片单独放在装配位姿上，MuJoCo 里 1 秒内倾斜 4.8°（Isaac 因刚体初始休眠看不出来）；`fragment_36d5ca6a07ba2b2f` 在摆放区落定时倾倒 18°。10-18 用户决定：锚点由夹具固定。已写进协议 `baselines.anchor = "fixture"`（`require_canonical` 核对）；`KinematicBackend` 不让夹爪抓走锚点；DexAgent 试验拒绝 `anchor_mode='free'`（`export` 仍可导出自由锚点做诊断）；ASPIRE 任务说明和 metadata 写明夹具。
- [x] Isaac 后端实现锚点夹具（锚点设为运动学刚体），其余碎片先落定再开始（`backend.start`）。
- [ ] 实物夹具：按锚点网格打印托架。
- [ ] 我们自己的 S2/S1/S0 走同一个 `AssemblyBackend` 和同一个评分，写同样的 `RunRecord`；S2 的任务上下文要带上任务的 `assembly_plan`（协议给所有方法）。
- [ ] ASPIRE 学习循环：benchmark profile 与任务上下文；计数 shim（拒绝留出种子和超出练习预算的 trial，记录 `sim_trials`）；成本计量（`cost_usd`/`llm_calls`，走共用预算池 `baselines/budget.py`）；每个 trial 的墙钟上限（被拒命令不耗步数，程序反复重试会永不结束）。上游的编码智能体是 Codex CLI 经 MCP 接 CAP 工具（`scripts/start_system2_agent.sh`），用 Claude 时需要：编码智能体外壳、沙箱外的计费中转（key 只在中转里）、带 GPU 的沙箱（程序与 Isaac 同进程）。
- [x] 实跑前提（10-19 用户决定）：DexAgent 与 ASPIRE 共用 100 美元模型预算池（`baselines/budget.py`，`baselines.llm_budget`；无每基线/每试验上限，失败重试都计入，不追加；flock 下先预留最坏值再结算，死进程的预留按最坏值计）；模型写的代码在 bwrap 沙箱里跑（`baselines/sandbox.py`；DexAgent 的 Stage 3 与交付复核；无网络、无家目录、只有试验目录可写；`baselines sandbox-check` 自检）。
- [ ] DexAgent 首次实跑（practice 种子 101，一个）：报告实际费用、模型调用数、耗时、失败原因，再定剩余额度；核实 `drop_block` 的丢弃情况，并在真实交付上验证轨迹导入（目前只在合成录制和官方示例上测过）。需要 key（`~/.config/anthropic.env` 目前装不下一个 key）。
- [ ] DexAgent 的可达性：在种子 101 下，`object_reach_check` 只给 7 片里的 3 片找到位姿，DexAgent 可能需要移动机械臂工位。另外每次运行时构建约 110 s（502 个盒子、19,443 个手-物接触对）；用集合覆盖可以把盒子减到 399 个，但需要显式惯量。
- [ ] 公平性：所有方法的目标来源与观测位姿来源必须一致，并随结果报告；正式评估前冻结种子分段，之后不再改。
- [ ] 位置误差的度量点：目前量碎片坐标系原点。这些 layout 里原点在装配底面，绕包围盒中心转 3° 会让原点移动 1.07–2.96 mm（8 片各不相同），旋转误差占用位置预算的程度差了约 3 倍。需与 S2 的 place 判据一起决定是改量包围盒中心还是顶点误差。
- [ ] 评估前登记冻结变体（在 selection 上选定）。占用目录让每个碰过留出种子的变体都可查，但目前不阻止临时登记新变体。
- [ ] DexAgent 的 `release_settle`（0.3 s、4 mm / 10°）与我们的判据（1 s、3 mm / 3°）不一致：要么对齐，要么作为混杂因素报告。
- [ ] GARF 候选目标的任务需要同时携带真值目标用于评分（目前直接拒绝评分）。
- [ ] **DexAgent 移植到同一套硬件**（10-19 用户决定「用一套硬件」「1移植」；进度表见 `baselines/dexagent/README.md`「Port to the shared hardware」；全部未提交）：
  - [x] 工位 `bench/`：CRX MJCF（S0 的关节链，质量=碰撞凸包×1000 kg/m³，暂定）、YAM 作为 DexAgent 的手（滑动指、指尖碰撞块保留贯通窗口）、场景与 setup.yaml（rig=世界下移到桌面）。
  - [x] 补丁 `patches/0001-crx-yam-port.patch`（21 文件）：关节位置伺服+重力补偿、20 N / 0.1 m/s 手指、滑动单位、对称钳口、CRX/S0 限速、接触带减速、门禁里加 `s0_delivery_check`。
  - [x] 执行：`executor.py`（60 Hz 斜坡、按阶段标签进带前声明接触窗口、拒绝不重定时、拒绝后也统一松爪静置评分）、`execute_trial.py`（每个交付只执行一次；Isaac 落定起点与 DexAgent 规划起点差 >1 mm / 1° 不执行）、CLI `execute`、`--bench crx_yam`、`score` 读试验自己的 S0 限值。
  - [x] Isaac 落定起点：种子 101 上 Isaac 与 MuJoCo 落定一致（最大 0.09 mm / 0.17°，倾倒的那片两边都倒 18°），不必用 Isaac 起点重新导出，执行时核对。
  - [x] 夹取：平行夹爪的对捏改为沿物体实际材料的「弦」（每 15°、几个横向位置，离质心最近的先），种子 101 的 7 片每片 20–24 个候选、前两个都能抓起（原来红色片 0 个）。
  - [x] 门禁分诊（脚本橙色片完整跑完 55 项）：机器人/手臂/手部检查全过；查出两个导出 bug（锚点写成 released、可视网格不水密，原生工位同样中招）已修；其余是脚本规划质量（放下压桌 17.7 N 尖峰、只做了 2/7 子目标）。
  - [x] 独立审查（1 HIGH / 2 MEDIUM / 5 LOW）全部修好并补测试：执行开始标记+出错补记录+留出种子必须带记录；进带处另起一块用当时观测声明窗口；块在速度变化最小处切；被拒的块不改窗口；锚点不开窗口；验收门从 home 位起步；补丁里验收门的描述。
  - [ ] 工具库分诊（第 4 步）余下：放到目标位会碰固定锚点；一条完整的多片脚本交付过全部门禁。
  - [ ] MuJoCo ↔ Isaac 标定（第 6 步）；mock 端到端（第 7 步）；实跑要先定预算和隔离主机。
  - 注意：S0 限值仍是测试值，共享硬件上的记录一律 `test_only`。完整 DexAgent 流水线（Stage 1/2）需要真人视频，不在此列。
- [x] **S0 首段加速度口径（所有方法；10-19 用户同意改，上限不变）**：实测速度语义——运动学替身与回放按构造报段速度；Isaac 的 PhysX 速度就是刚执行完的指令段速度（关节 1 匀加速斜坡 RMS 差 0.3 mrad/s，不是平滑运动的周期末速度 +a·dt/2；阶跃第一个周期就报出）⇒ 都是上一周期平均。实现：`RobotSnapshot.velocity_window_s`（后端必须声明）、`check_joint_path(qd_window_s=...)`：首段间隔 = (窗口 + 首段)/2（周期平均：与之后各节点同读法；瞬时速度窗口 0：仍为 2Δv/dt；不声明或非法即拒绝）；ASPIRE 速度上限随之翻倍（60 Hz、TEST 20 rad/s²：0.15→0.30 rad/s）；DexAgent 30 fps 斜坡的 8 rad/s² 规划上限不变（帧内速度变化落在一个伺服周期，与口径无关）。测试：切成两段与整段读数相同、恒定加速跨块边界不再被误拒；变异 9 处 8 处抓到，1 处等价（`sim/pick.py` 传 0：其规划器按更严读法留余量）。历史判定：无变化——新读法从不更严，记录里没有因加速度被拒的指令（ASPIRE Isaac 运行 26 次调用、各路径首步旧读法最大 17.98、新 8.99 rad/s²；DexAgent 交付全部执行；P2 抓放无拒绝）；唯一被拒的是 ASPIRE 收尾静置的速度 16.7 rad/s（速度检查，不受影响）。真机 FANUC 后端写时自行声明窗口。
- [x] **碎片按 8 mm 体素互锁，原任务装不成（10-19 发现，影响所有方法；10-20 定案，见本条最后）**：碎片几何是 8 mm 体素，相邻碎片的接缝是一格一格的台阶，彼此勾住约一格。按直线运动、不转动分析：体素格上精确判定（`experiments/assembly_paths.py`），只有锚点在场时，红、绿、青、橙没有任何一个格子方向可走；8 种锚点逐一算过，没有一种能按某个顺序全部拆开。MuJoCo 中（`experiments/assembly_depths.py`，用 DexAgent 导出的碰撞盒、取仿真自身产生的接触）这四片最好的出路仍要穿入锚点：红 3.55、青 3.55、橙 3.76、绿 10.0 mm。⚠️ 先前报的 7.0–10.0 mm 来自 `mj_geomDistance`：两个盒子只共一条棱时它也会报几毫米穿透，已改用接触深度。
  - **10-20 候选修复，已验证；用户 10-20 选红作锚点**（任务文件尚未替换：先定装配计划是否写进任务、是否给所有方法，再一次换掉 `runs/baseline_tasks/` 和 DexAgent 示范）：`prepare --straight-assembly-anchor`（`interlock_repair.py`）只改接缝体素的归属，不增删体素，最后把活动碎片的孤立立方体交给不挡顺序的邻片。只有红、白两片贴桌面，两种锚点都试了：
    - **红作锚点（底座）**：换了归属的立方体 62/2116。蓝、绿、黄、品红、白（头盖，最后放）从上方放入，橙、青水平推入。修复算法原来每步只挑当下最便宜的一片，得到 92 格、白第二个放；现在每一片都试作最先拆下的那片，取总改动最少的方案（白锚点结果不变）。方向不变时有 36 种无碰撞顺序。逐片放下静置：Isaac 里 36 种的每一步都不动（≤ 0.0004 mm / 0.0007°，`runs/assembly_paths/settle_red_plan_isaac.json`）；MuJoCo 里（机器人停在 keyframe 0）全部在容差内，只有蓝先于橙放时蓝滑 1.9 mm（`settle_red_plan_orders.json`）。⚠️ 先前记的「只有 6 种全程不动、绿先于黄放时青被压歪 3.9 mm / 6.3°」来自 scratch 检查让机器人从 qpos0 起步（手在头骨里、压进桌面，不接触碎片），作废；`settle_red_orders.json` 是那次的结果。计划顺序：橙、蓝、绿、黄、青、品红、白，两个引擎里每一步都不动。机器人要搬整块头盖（约 0.47 kg）。产物在 `runs/skull_straight_red/`、`runs/baseline_tasks_straight_red/`；旧版在 `_v1`。
    - **白作锚点（另一个候选）**：换了归属的立方体 106 个（先前写的「108」是翻转次数），7 片全部水平推入，重力下无位移。产物在 `runs/skull_straight/`、`runs/baseline_tasks_straight/`。
    - 两种都已验证：各片连通；体素格判定 7/7 可拆；MuJoCo 沿顺序接触深度为 0；DexAgent 导出检查通过（种子 101）。
    - 无论选哪种，装配顺序都是固定的，且都有水平插入，DexAgent 导出的示范要改。
    - 改几何就是改任务，必须在任何方法看到 held-out 之前定。
    - **原始网格碎片（不体素化）也装不成（10-20 查）**：8 片是封闭实体，头骨是实心的，GT 接缝零间隙。0.5 mm 实体体素、约 1,100 个上半球方向及反向：每片缩 0.5 mm 时，橙、黄、青、绿各自和白往任何方向都分不开，任何锚点都没有直线顺序；缩 1 mm 时绿仍被白包住，最好的方向也要穿过 1.4 cm³ 的白（绿共 19 cm³）；缩 2 mm 仍锁（0.27 cm³），要缩到 4 mm（接缝 8 mm）才分得开。体素互锁不全是体素化造成的；改用原始碎片省不掉装配计划，还要处理凹形碰撞。代码 `experiments/mesh_paths.py`；产物 `runs/assembly_paths/mesh/`（0.5、1 mm）、`mesh_wide/`（2、4 mm），剖面图 `mesh/green_slices.png`。
    
    可视化页面：claude.ai artifact H1sBNuUK2Nc9yM2C13E12b。
  - **10-20 定案（用户确认）**：红作锚点；装配计划是 GT 的一部分，主评测给所有方法看。已完成：
    - `prepare --assembly-order`：按修复给出的方向在体素格上精确判定，能直线放入才接受，写进 `ownership_repair.plan`。几何在 `runs/skull_red_plan/`：网格与 `runs/skull_straight_red/` 逐字节相同，只多了计划。
    - `AssemblyTask.assembly_plan`（世界坐标系的插入方向）；协议 `baselines.assembly_plan = "given"`；`verify_task` 拒绝没有计划的任务。
    - 任务换成 `runs/baseline_tasks/`（基础任务、种子 101；id `view_gt-00f0cc2853e9-anchor-fragment_038755611176a8c9`）；旧任务归档到 `runs/baseline_tasks_diag_white/`。
    - ASPIRE 任务上下文加了 `metadata.assembly_plan` 和一句说明。
    - DexAgent 导出：子目标按计划排序并写明最后一段直线移动；替身视频最后沿插入方向直线进入；`stage3.notes` 写明顺序和方向。
    - 重力检查搬进仓库：MuJoCo 版 `experiments/assembly_settle.py`（机器人停在 keyframe 0 并每步钳住，与 DexAgent 自己的落定一致）、Isaac 版 `experiments/assembly_settle_isaac.py`。结果见上一条：顺序只受碰撞约束。
    - 独立代码审查（1 HIGH / 1 MEDIUM / 若干 LOW）已处理：执行器测试夹具缺计划（HIGH）；建任务时在 layout 自己的体素格上复核计划（MEDIUM）；协议值只认 `given`；重力检查从 keyframe 起步、核对 layout 的 sha256；`export_task` 也转绝对路径；按工位用词。变异检查 32/32 抓到。
- [ ] 可选消融（以后再做）：不给计划，只给所有方法同样的检查工具（直线插入检查、重力静置），记录调用次数与费用；DexAgent 要另加规划器，必须标为改过的基线。
- [ ] **10-19 用户决定，待实施**：S0 余量与接触三参数只作仿真开发候选值，与加速度、TCP 速度上限一起经正常抓放验证后给最终参数表再冻结，不靠改 `test_only` 标签认定有效；步数预算实测单片与完整 7 片耗时、留余量、加墙钟上限（只用 practice 种子）；手指 0.1 m/s / 20 N / 15 mm 转接板标为假设，正式评估前冻结，实测不同=新协议版本；位置误差改量碎片局部包围盒中心（旋转单独判；S2 放置判据与评分器同步改）；静置与成功判据尽量统一为 1 s、3 mm / 3°（重新验证，上游门禁对不齐要报告差异）；起始会倾倒的碎片调为稳定摆放，所有方法用同一套落定后的初始条件，held-out 前冻结。held-out 暂不开放。

验收：每个方法的留出成绩都来自冻结的变体，每次 trial 都有记录，评分相同；结果附上费用、仿真次数和混杂因素。

## 本轮进展与 GARF 工具集成

- [x] 推送 jing 分支；保留上游基线 df03da2fc26a47d0dbf25782ec04fb6fc2ffc776。
- [x] 实现 assembled-GT GLB 入口、稳定节点 ID、至少六片校验。
- [x] 接通共同体素网格；增加保留全部组件选项、六连通与闭合检查。
- [x] 导出 NPZ、毫米 STL、来源哈希和变换 manifest；合成输入 CPU 回归测试。
- [x] 接通现有双臂三相机场景 launch，要求显式允许 Robotiq proxy。
- [x] 定位并读取真实八片 Hoolock 头骨；记录原 bundle 是运动学演示。
- [ ] 完成真实头骨可打印拓扑与物理装配验收。
- [ ] 在 Linux/NVIDIA Isaac Sim 环境完成 launch 与物理抓放验收；本机 CPU 测试不代表通过。
- [ ] 解决表面体素非流形案例、实体填充/薄壁、装配间隙与制造试样；禁止静默丢片。
- [x] 核对用户 GARF.ckpt 的 Drive 文件名/大小，写明工具与 S2/S1/S0 的职责。
- [x] 下载权重到仓库外，记录 SHA256、上游代码 commit；独立官方 CUDA 环境严格加载通过。
- [ ] 核实权重再分发许可；当前仓库不包含权重。
- [x] 核实 GARF checkpoint 参数键与网络结构；真实八片头骨推理通过，精度尚未达装配要求。
- [x] propose_assembly 调用链：固定 ID、点云预处理、反归一化、候选、失败/超时、缓存和版本（JSON-lines runner + `s2/garf_client.py` 的截止时间、杀进程重启和按几何内容的缓存；没有网络服务）。
- [x] 结果 schema、SE(3) 校验、工作台坐标恢复（`s2/candidates.py`）、重叠/接触筛选；评分未经校准不当概率。
- [ ] 候选目标的可达性检查（用 S0 的 IK 与胶囊体检查预筛）。
- [ ] 实现三相机到已知扫描片身份与位姿估计，或有尺度 3D 重建；标记模拟 oracle 模式。
- [x] S2 tool-calling 与编排状态机：observe → GARF → geometry validation → skill → 执行 → verify/recover（`s2/orchestrator.py`，替身执行器 + 真实 S0 的跨层测试）。
- [ ] 接入 Isaac 中的真实 S1/S0 执行器和放手后核验器；已验证簇的融合几何重新注册到 GARF runner。执行器要把 `S1Runtime.trained_combinations` 交给编排器（默认 None 只按技能把关），并把 S1 的 `ValueError`（未训练组合、缓冲不足）和 `ActionRejected` 记为 `ExecutionReport('rejected')`；漏掉的话异常会让整轮以 `error` 结束。S0 接受动作块后要调用 `S1Runtime.contact_memory.accepted(chunk)`，每个 episode 开始时调用 `reset()`，否则 retreat 拿不到窗口。
- [ ] 数据记录 GARF 候选与 S2 决策；比较 GT/GARF/GARF+恢复，评估分布偏移。

## GPU 验证与逐步 GARF（本轮）

- [x] 完成真实头骨 GARF 单次推理、GT 离线误差报告；输入点云各自随机局部坐标，GT 单独保存。
- [x] 实现多 seed 候选工具，固定输入采样、仅改变噪声；最多八个候选，不默认选中。
- [x] 实现已验证刚体组合的点云/法向融合及成员位姿展开；CPU 测试通过。
- [ ] 真实闭环验证“安装→稳定性判断→合并→重推理”，剔除内部接触面并量化分布偏移。
- [ ] 实现不依赖 GT 的候选筛选：穿透、接触、已装配一致性、IK/运动路径、观测证据。
- [x] 实现 RGB-D 光学 Z 反投影、已知初始位姿下局部 ICP、锚点坐标恢复。
- [ ] 实现分割、实例身份、全局粗配准、对称性/遮挡检测与相机标定回归。
- [x] Isaac Sim 5.1 真实八片诊断场景运行 120 次 world.step；同一暂停时刻采集三路 RGB-D。
- [ ] 修复非流形/不连通体素；当前诊断模式不导出打印件，不能用于物理装配成功验收。
- [ ] 修复原 FANUC 资产部分 link 的质量/惯量警告，校准动力学。
- [x] 候选工具的超时/杀进程重启/缓存（`s2/garf_client.py`）。真实 S2 API 调用见 P0。

## GARF 官方对照审计（2026-10-14，详见 reports/garf_audit）

- [x] 复用官方 MeshInferenceDataset 与 scheduler，检查坐标回代、四元数顺序、尺度。
- [x] 检查原始八片闭合/绕序/正体积；测试统一翻转法向。
- [x] 对比 SDPA/Flash、exponential/linear+one-step；保留全部失败报告。
- [x] 固定 trimesh 采样 RNG，测试5000与10000点、提高小片最低点数。
- [x] 用官方瓶子样例做对照：多数碎片可恢复，尚非全件成功。
- [x] 测试4个noise的无GT表面穿透/接触筛选，全部拒绝；GT仅校准检查。
- [x] 添加完全重合实体的内移探针回归检查、S2输出拒绝状态与mesh注册表。
- [x] 运行真实头骨的理想两片合并消融，单独报告剩余六片误差；未改善到可装配精度。
- [x] 当前“内移探针+表面”版本已完成历史4候选复测：全部拒绝，GT几何检查通过；新缓存特征候选另列待测。
- [ ] 统一生产点云适配器与官方mesh输入的模型尺度、面积预算和采样；不凭毫米单位猜模型尺度。
- [ ] 在明确训练分布、头骨fracture segmentation表现后，定位领域差异/输入几何的具体根因。
- [ ] 测试融合后内部面删除、外表面重采样与多种锚点；不能用GT认证实际安装稳定性。
- [ ] 出现合格候选后再推进IK/路径/物理夹持，禁止用最低分失败候选冒充装配成功。

- [x] 发现并隔离 eval 模式 PointTransformer 的随机序列分组，固定 encoder seed；候选记录特征指纹。
- [x] 固定采样和编码器后重跑官方对照：仍有82.63–91.47 mm误差，未把随机源修复误报为精度修复。

- [x] 进一步发现固定seed不保证编码结果逐字节一致；增加单批几何特征缓存，GPU验证4候选和合并实验各自特征指纹一致。
- [ ] 对缓存特征后的新4候选做完整几何筛选；历史4候选的筛选结果不可直接套用。
