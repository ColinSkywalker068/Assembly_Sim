# 分层接口与数据约定

S2→S1、S1→S0 两个契约已实现在 `contracts/`（`skills.py`、`actions.py`），S2 编排在 `s2/`。真实执行器（Isaac 中的 S1 推理 + S0 下发）尚未接通，下文明确区分“已实现”和“设计”。

## 坐标与时间

米、秒、弧度；世界 Z 向上。`T_A_B` 将 B 坐标变换到 A。原始 pose 用 `[x,y,z,qw,qx,qy,qz]`，存储单位四元数。记录 world/base_left/base_right/flange/TCP/camera 外参及版本。相机光学轴转换显式保存，不能混用 USD 与 OpenCV 轴。

所有流保存采样时间、接收时间和时钟域，转换到 episode 单调时间轴。同步图像按曝光/采样时刻匹配实际状态；不得将执行后状态与刚执行动作直接当成下一步监督对。

## 统一的 S2 程序入口

默认 `s2.planner=aspire_program`、`execution_mode=skills`，实现于 `s2/aspire.py`。
ASPIRE 的 Python 程序通过白名单工具读取当前观察、请求 GARF 候选并提交一个技能请求；
`AspirePlanner.plan(context)` 返回现有 `PlanOutcome`，编排器照常复核后调用 S1/S0。
程序本身不能调用底层运动 API。提交之后任何工具调用或程序错误都丢弃此次请求。

`execution_mode=direct` 则使用同一个上游运行器注册 CapApi，走 ASPIRE 适配器的关节路径检查，
不经过 S1。该模式使用原 ASPIRE trial runner 的种子/评分/记录协议，结果归 `method=aspire`。
两模式不能混合注册工具。旧 `ClaudePlanner` 仅在显式 `planner=claude_tools` 时使用；
下文 Claude 客户端细节描述该兼容路径，不是默认 S2。

程序生成/修正循环、运行隔离与硬超时、真实 Isaac/S1 执行器仍未完成。

## S2 -> S1

SkillRequest 包含 request_id、skill_id、fragment_id、active_arms、target_frame、target_pose、support_fragment_id、completion_predicate、timeout_s。技能候选为 pick/place/align/insert/support/retreat/reobserve；模型只能选择已经训练和实现的技能。target 必须来自可观测/预测几何或明确给定条件，记录 provenance。初期可人工提供请求。

已实现（`contracts/skills.py`）：

| skill | 臂数 | fragment | 目标位姿 | 允许的完成判据 |
|---|---|---|---|---|
| pick | 1 | 必需（且须带观测位姿） | 无 | grasp_held |
| place | 1 | 必需 | 必需 | pose_within / released_stable |
| align | 1–2 | 必需 | 必需 | pose_within |
| insert | 1–2 | 必需 | 必需 | pose_within / released_stable |
| support | 1 | 必需（且须带观测位姿） | 无 | grasp_held |
| retreat | 1–2 | 无 | 无 | retreated |
| reobserve | 0 | 无 | 无 | observed |

- `SKILL_IDS` 的顺序固定，S1 以其下标作为技能条件；改顺序等于换条件编码，旧 checkpoint 失效。
- `validate_skill_request` 拒绝：过期 observation、未训练的运动技能（给定 `trained_combinations` 时还包括未训练的手臂组合）、臂数/fragment 不符、未知 fragment、support 与 fragment 相同、非 world 坐标系、非刚体目标、超出范围的完成参数、`timeout_s` 不在 (0, 120]。
- `fragment_pose` 是该碎片在本次 observation 中观测到的当前位姿 `world_T_fragment`，由编排器从 observation 复制（来源 `observed_geometry`，并记录 `pose_source`；仿真真值记为 `oracle_sim`），让 S1 知道要处理哪一块。pick/support 没有它就被拒绝（应先 reobserve）；place/align/insert 在碎片可见时携带；retreat/reobserve 不允许携带。编排器执行前核对它与当前 observation 逐位相同。
- `active_arms` 由编排器规范为 ARMS 顺序（left 在前），S1 的条件编码依赖这个顺序。
- 训练过的手臂组合：S1 checkpoint 记录训练窗口数达到下限的 (skill, arms) 组合（`S1Runtime.trained_combinations`，arms 为 ARMS 顺序），其他组合 S1 一律拒绝。把它和 `trained_skills` 一起交给编排器（`AssemblyOrchestrator(..., trained_combinations=...)`）后，规划解析和执行前复核都会拒绝未训练的组合，例如只训练过左臂 place 时请求右臂 place。`trained_combination_set` 要求每个组合是运动技能、arms 为 ARMS 顺序且臂数符合上表，并且技能集合恰好等于 `trained_skills`；Claude 规划器的状态 JSON 会多一项 `trained_arm_sets`。不传时只按技能把关，用于没有 checkpoint 的脚本运行。
- 目标位姿是 `world_T_fragment_goal`，由编排器从记录中解析，S2 只引用 `candidate_id`，不抄写数字。GARF 目标的 provenance 必须含同一 observation、`candidate_id`、`checkpoint_sha256`，且该候选几何筛选为 `needs_motion_validation`。
- 完成判据和超时取自 `configs/jing.json` 的 `s2.completion_defaults` / `s2.timeouts_s`（目前是待打印件试配后替换的临时设计值），S2 不能自定容差。
- 所有字段在构造后只读（映射为 `MappingProxyType`，数组不可写）。

## S1 -> S0

ActionChunk 包含 request_id、chunk_id、observation_time、valid_until、左右 anchor 的 T_world_tcp、严格递增的 waypoint_times、20D waypoint 数组、控制模式，以及可选的接触声明 `contacts`（每只手最多一个 `ContactWindow`）。

数组顺序固定为 left(position3, rotation6, width1), right(position3, rotation6, width1)。每手每个 waypoint 表示 `T_anchor_target = inverse(T_world_anchor) @ T_world_target`。anchor 在整块预测期间保持固定；宽度为绝对米值。S0 用 `T_world_anchor @ T_anchor_target` 还原绝对目标。

rotation6 先放旋转矩阵第一列三个数，再放第二列三个数。解码：b1=normalize(a1), b2=normalize(a2-dot(b1,a2)b1), b3=cross(b1,b2)。拒绝非有限、零长度或近共线输入，不能默默生成任意姿态。非活动臂也必须有明确 hold 目标。

S0 返回 accepted/rejected/holding/completed/failed、reason、measured_tcp、measured_width、tracking_error、timestamp。高层失败与动作完成不等于装配成功；装配由独立 evaluator 判断。碰撞检测包含 arm-arm、arm-object、tool-object、table；预期接触使用明确接触对象/阶段，不关闭全场碰撞。

已实现（`contracts/actions.py`）的具体约定：

- 时间一律是 episode 单调时钟上的绝对秒。`waypoint_times` 严格递增且第一个严格晚于 `observation_time`；`valid_until >= observation_time`。
- `encode_chunk` 把各手的世界目标 `(N,4,4)` 和宽度 `(N,)` 编成 `(N,20)`；`decode_chunk` 还原，二者互逆（测试误差 1e-12）。
- 构造时即拒绝：非刚体 anchor、缺任一臂 anchor、重复或未知的 `active_arms`、非有限值、负宽度、rotation6 零长度列或近共线列（`|a1×a2|/(|a1||a2|) < 1e-3`，约 0.06°）。
- `executable_reason(chunk, now, max_observation_age_s=...)` 给出 S0 必须拒绝的原因：当前时间早于 `observation_time`（时钟域不一致）、观测超过 S0 自己的年龄上限（不依赖 S1 自报的 `valid_until`）、`now > valid_until`、所有 waypoint 已过、非活动臂移动（平移 > 0.1 mm、转角 > 1 mrad 或宽度变化 > 0.1 mm）。返回 None 才可执行。
- `encode_chunk` 拒绝非刚体目标，不做静默正交化。
- `ContactWindow(arm, kind, center_xy, fragment_id)`：
  - `kind` 只有 grasp / place 两种，`fragment_id` 必填；
  - `center_xy` 是世界坐标的 x、y（米，有限值，内部存成两个浮点数）；
  - 同一块里每只手最多一个。
- `contracts.skills.contact_windows(request, previous=None)` 按技能给出声明，每只活动手一个：
  - pick / support 取碎片观测位姿；
  - place / align / insert 取目标位姿；
  - retreat 不单独声明，沿用每只活动手上一次被 S0 接受的抓 / 放窗口（`previous`，手 → `ContactWindow`），没有的手就没有声明；
  - reobserve 没有声明。

  这样 retreat 只能在这只手上一次被 S0 接受的抓 / 放窗口里贴桌（在窗口里也可以重新下降），不能拿它在任意位置贴桌。
- `contracts.skills.ContactMemory` 保存每只手上一次被 S0 接受的窗口：
  - `windows(request)` 就是带着这份记忆的 `contact_windows`；
  - `accepted(chunk)` 由执行器在 S0 接受动作块后调用；只是声明过、被拒或没执行的窗口不会被记住；
  - `reset()` 在每个 episode 开始时调用，因为旧窗口在新布局里可能正压着别的碎片。

  只有 S1 运行时按这套规则产出声明，S0 本身不知道技能。以后其它产出方（ASPIRE 的接触、脚本专家、直接调用 `check_joint_path` 的代码）也必须从 S2 校验过的目标或碎片位姿推出窗口，不能用程序自己给的坐标。
- `S0Status` 的 rejected/failed 必须带 reason。动作块数组和 anchor（`MappingProxyType`）都是只读的。

S1 产出动作块的约定（`s1/chunking.py`）：第 k 个航点（从 1 起）记为 `observation_time + k/10`，即训练时该位姿相对观测的时刻；推理和传输延迟不靠改写时间吸收，而是由 S0 丢弃已过去的航点、拒绝超过 `valid_until` 的块。S1 默认 `valid_until = observation_time + 0.25 s`。非活动臂由 S1 直接写成精确保持（位姿为 anchor，开口为调用方提供的上一次指令开口，否则会被 S0 拒绝）。预测出退化旋转时 S1 抛出 `ActionRejected`，执行器必须把它记为失败步骤（`ExecutionReport('rejected')`）并让 S2 重新观察，绝不当作动作执行。S1 只把开口裁到 ≥ 0，真实开口范围由 S0 负责。

S1 运行时（`s1/runtime.py`）：`S1Runtime(checkpoint, *, device, seed, stale_after_s).chunk(request, tcp, cameras, *, commanded_widths, observation_time=None, chunk_id=0)` 返回 `(ActionChunk, info)`。`commanded_widths` 必填，作为被保持手臂的开口。`observation_time` 默认取两路 TCP 缓冲都已到达的最新时刻，比它晚就拒绝；未训练的 (skill, arms) 组合也拒绝，两者都抛 `ValueError`，执行器同样要记为 `ExecutionReport('rejected')`。条件编码第 2 版：技能下标、ARMS 顺序的活动臂掩码、目标位姿与碎片观测位姿（都相对第一只活动手的 anchor，各 9 维加掩码）。checkpoint 记录 `SKILL_IDS`、ARMS、相机、时域和版本号，不一致就拒绝加载。采样噪声由 `PolicyConfig.sampler_eta` 决定：默认 0 为确定性 DDIM，1 对应 DDPM 后验方差；该值随 checkpoint 保存。每个动作块都附上 `S1Runtime.contact_memory.windows(request)`：抓 / 放窗口来自请求自己的位姿，retreat 的窗口来自记忆。运行时只读不写这份记忆：执行器在 S0 接受动作块后调用 `contact_memory.accepted(chunk)`，每个 episode 开始时调用 `contact_memory.reset()`。执行器不回报时，retreat 没有声明，贴桌照常被拒。

S0 执行器（`s0/adapter.py`）在契约之外还要求：

- `plan(chunk, now=, joints=, joint_velocities=, widths=, commanded_widths=, tcp_at_observation=)`：`tcp_at_observation` 是调用方从状态历史取出的观测时刻实测 TCP，两臂 anchor 偏离它超过 `S0Limits` 的位置/转角容差即拒绝。`commanded_widths` 是每只手上一次下发/接受的开口：位置控制夹爪的夹紧力由指令决定（指令 0.020 m、被碎片卡在 0.030 m 时，按 0.030 m“保持”就会松开），所以保持、制动、非活动臂的开口检查和活动臂开口插值的起点都用它；实测开口 `widths` 只用于状态上报。
- 同一 `request_id` 的 `chunk_id` 必须严格递增，重放或乱序的块被拒绝。
- 任何拒绝都返回**制动轨迹**：每条臂各关节以同一时刻停下、减速度不超过加速度上限，开口保持上一次指令值（绝不松开），之后保持；运动中的非活动臂同样制动。被接受的块还要满足“从末状态 (q, qd) 按加速度上限制动后仍在限位内”。实测状态无效时返回 `('failed', None)`，调用方必须急停。
- 桌面检查包含夹爪：工具包围盒（手指 0–0.8 rad 扫掠包络，x 最远 0.1628 m，超出代理 TCP 27.8 mm）的 8 个角点都要高于桌面加 `table_clearance_m`，TCP 不低于 `tcp_min_height_m`。
- **接触阶段**：只有某只手在某个采样点违反上面两条之一时，S0 才去看这只手声明的 `ContactWindow`，以下条件全部满足才放行：
  - TCP，以及手指包围盒低于常规底线的全部部分（低于底线的角点，加上各条棱与底线平面的交点），x、y 都在声明中心的 `contact_region_radius_m` 之内；
  - 手指包围盒和 TCP 都高于桌面加 `contact_table_clearance_m`（必须大于 0）；
  - 夹爪本体包围盒仍高于桌面加 `table_clearance_m`；
  - 与这些采样点相邻的伺服区间都不超过一个伺服周期（允许 1% 误差），区间内手指包围盒每个角点的速度不超过 `contact_max_speed_m_s`；实测起点本身就需要接触时，第一个区间也算在内；采样时间必须严格递增。

  S0Adapter 还记住每个请求第一次被接受的接触：同一请求里，接触的类型或中心一变就拒绝。声明按 `ContactWindow` 基类重新校验，伪造的声明只会得到拒绝和制动。

  请求 id 必须唯一：S0 的记忆和块序检查都按 id 记。代理夹爪的手指盒是全行程包络，朝下时底角离 TCP 轴 78 mm，所以 `contact_region_radius_m` 至少要大于这个值才能抓取。

  手臂停在接触带里、记忆里却没有它的窗口时（例如重启之后），S0 会拒绝所有贴桌的动作，retreat 也离不开，需要人工恢复。

  照常拒绝的情况：没有声明、接触参数未配置、缺手指几何、超出区域。另一只手、连杆和两臂间隙都不受影响。

  接触带以外的接近速度只受关节限制：规划好的减速由 S0 的加速度检查保证，实际跟踪的超调由控制器负责。

  直接下发关节目标的 `check_joint_path(..., contacts=...)` 用同一规则，但目标必须按伺服频率给出，而且它不记忆请求。

  `contact_windows` 只给活动手生成声明，所以"一只手在桌面上扶持、另一只手放置"这类动作目前会被拒绝。

  工具盒拆成本体（base_link，x 最远 89.8 mm、厚 75 mm）和手指（x 44.6–162.8 mm、厚 35 mm）两部分，两者合起来正好是工具盒（`crx10ial_kinematics.json` 中的 `parts`）。碎片仍不是 S0 的碰撞对象。检验脚本是 `experiments/s0_contact_check.py`，结果见 `reports/s0_contact_2026-10-18.json`。
- YAM `linear_4310` 的几何来自官方模型（`core/robots/linear_4310.py`）：`s0.kinematics.yam_model` 把它装在暂定的 15 mm 转接板上，TCP 沿用代理的约定（TCP 坐标系 = 法兰坐标系平移到抓取点，+x 接近，手指沿 y 开合），工具盒拆成本体加转接板与两指全行程两部分；开口范围 `yam_width_range_m()` = (0, 93.96 mm)。
- 关节加速度上限、实测 YAM 开口与 `flange_T_tcp`（要等转接板）、各类间隙、接触阶段的三个参数在 `jing.json` 中仍为 null，必须显式提供；测试里的 20 rad/s² 只是测试值。

## Episode

建议三种 source：sim_expert / umi / robot_teleop。共同字段：

- episode_id、task_instruction、skill_segments、fragment_ids、geometry_hash、calibration_version、seed、source。
- images.left_wrist/right_wrist/agent 与各自 timestamps、可用性 mask。
- state.tcp_pose_left/right、gripper_width_left/right、inter_tcp_pose。
- action.requested_tcp/width、accepted_tcp/width、issue_time、scheduled_time；UMI 示范为未来实际 TCP 轨迹，另记 action_origin，不冒充机器人命令。
- diagnostics.joint_positions/velocities（UMI 缺失）、tracking、接触/力（若有）。
- privileged.gt_object_poses（仅仿真专家/评估）、末态误差、success、failure_reason。

先保存高频原始数据，再按 10 Hz 策略时间网格插值状态和对齐图像，旋转用 SLERP。保留完整 episode，不跨 episode/技能边界错误拼接动作块。训练可分别评估 command-target 与 achieved-trajectory 监督，不能未经标记混合二者。

S1 默认不读取 privileged 或 joint diagnostics，确保仿真/UMI 输入一致。训练集统计独立保存并复用于推理。不能随机切相邻帧造成泄漏。

S1 的实现为 episode schema 2（`s1/episode.py`，schema 1 拒绝加载），新增 `initial_condition_id`、相机流的 `receive_times` 与 `clock_domain`、技能段的 `fragment_pose` 与 `fragment_pose_provenance`。数据规则（`s1/dataset.py`）：

- 默认只用 `success is True` 的 episode，显式开 `include_failures` 才用失败示范；目标来源为 `oracle_ground_truth` 或碎片位姿来源为 `oracle_sim` 时报错，除非 `allow_oracle`。
- 状态严格因果，取 t 时刻或之前最新的样本；相机帧在 `receive_time` 之后才可用，帧龄从曝光时刻算；指令流时间戳记为该指令预期到达的时刻。
- 监督窗口一直走到技能段末，超出部分保持段内最后一步；历史只在录制开头补齐。监督跨过超过 0.1 s 的位姿流空洞、或状态旧于 0.1 s 的窗口丢弃并计数。
- 按 `initial_condition_id`（缺失时用几何哈希 + seed）分组划分集合，同一摆放的重试不会同时出现在训练集和测试集。

## 观测构建

S1 输入三视角历史 + 双 TCP/宽度 + 双手相对位姿 + 技能条件/目标；S2 使用关键帧、可观测场景图、CRAG 预测和反馈。第一版原始 DP 没有现成的语言大模型接口，采用明确 skill ID 与目标编码；语言到技能的映射由 S2 完成。

UMI 外部相机缺失时显式标记，需训练对应 missing-view 模式；不生成假的同步第三视角。相机定位误差、夹爪几何差异和真实延迟属于 sim-to-real 验证项。

## RGB-D、完整 part 点云与 GARF

完整扫描点云定义每片的局部几何 `P_i`，需统一到米并提供一致定向法向量。没有法向量时必须估计并验证，不能填零。只有 XYZ 的点云可以先作为定位模板，但尚不满足此 GARF adapter 输入契约。

RGB 与 depth 必须已对齐、去畸变。对实例掩码内像素，光学 Z 深度反投影：

`p_camera = [(u-cx) z/fx, (v-cy) z/fy, z]`

`p_world = world_T_camera @ [p_camera, 1]`

腕相机外参随关节运动：`world_T_camera(t) = world_T_base @ base_T_flange(q(t)) @ flange_T_camera`；三路图像及关节状态必须按采样时间对齐。USD 相机轴转 OpenCV 光学轴使用 `diag(1,-1,-1,1)`，深度使用 `distance_to_image_plane`，不能把沿射线的距离当 Z。

**内参只能取渲染器实际使用的值。** K 采用 OpenCV 像素中心约定（整数 (u, v) 是像素中心，居中图像 cx = W/2 − 0.5、cy = H/2 − 0.5），与 `depth_to_world` 一致；外参取渲染器的 view 矩阵，USD 相机 prim 位姿只作对照（Fabric 下腕相机的 USD 位姿可能滞后）。 工作站相机只设置了焦距，USD 胶片背保持默认 20.955 × 15.2908 mm（宽高比 1.37），而输出是 1920×1440（4:3）。渲染器按图像宽高比推导竖直光圈，像素是方的：2026-10-18 在 Isaac Sim 5.1 实测，render product 投影矩阵给出 fx = fy，用同一帧的光学 Z 与射线距离做最小二乘 `(r/z)^2 - 1 = ((u-cx)/fx)^2 + ((v-cy)/fy)^2` 得到的 fx、fy 与之相差 < 1e-6。按 USD 光圈属性算出的 fy 偏大 2.782%，因此 `reports/2026-10-14.json` 里的 fy（2354.36 / 1695.14）是错的，fx、cx、cy 与外参无误（USD 相机位姿与渲染器 view 矩阵相差 1e-15）。`smoke.py` 现在从 `camera_params` 注释器取投影矩阵，并在两种来源相差 > 0.5% 时直接失败；证据见 `reports/2026-10-18_camera_intrinsics.json`。

实例分割/身份匹配 → 全局粗配准 → 局部 ICP → 重叠/误差/对称性检查 → `world_T_part_current`。已实现 `depth_to_world` 和需要初始位姿的 `refine_pose`；没有实现自动分割、全局初始化或身份匹配。ICP 残差低不代表身份正确，对称碎片尤其需要多视角验证。

GARF 输入完整局部几何，输出 `assembly_T_part_goal`。锚点观测恢复 `world_T_assembly` 后得到目标；当前位姿与目标之差用于规划。S1 可以继续吃 RGB+机器人状态，把 depth 用于几何工具；若 S1 直接吃 RGB-D，需要相应编码器与训练，不能把深度直接加到既有 RGB checkpoint。

## 多噪声与逐步重装配

`tools.garf.TOOL_SPEC` 与 `CANDIDATE_TOOL_SPEC` 可作为 Anthropic tools 定义。后者接受 `seeds`（1–8 个不重复整数），点采样固定 seed=42，噪声随 seed 改变；结果保留所有候选，无默认赢家。生产筛选必须基于观察与几何，离线 `evaluate.py` 才能读取 GT。

已稳定装配集合 C 的输入可以合并为一个刚体：

`P_C = union_i (cluster_T_i @ P_i)`

`assembly_T_i = assembly_T_C @ cluster_T_i`

`fuse_verified_cluster` 要求外部独立确认稳定性，不能用 GARF 的预测充当“已经装好”。工具返回成员映射，S2 保留原始 fragment ID；`expand_cluster_pose` 恢复每片预测。若接触松动、位姿漂移或遮挡导致不确定，应拆回独立状态并重新观察。当前融合保留内部接触面，属于分布偏移实验；表面去重/遮蔽及真实闭环融合尚未验证。

建议状态机：observe → sample_candidates → validate/rank → plan_skill → execute → release_and_verify → merge_verified_cluster → reobserve。每次只执行已验证的下一步；场景变化使旧候选失效。相同已知几何的候选可缓存，但锚点位姿和可达性必须重新计算。超时/取消由调用进程管理；当前 JSON-lines runner 尚无网络服务或内部抢占式超时。

## S2 工具与编排（已实现，尚未接真实执行器）

- **给模型的工具**（`s2/planner.py`）：`propose_assembly_candidates` 与 `submit_skill_request`，均为 `strict: true`。strict 模式不支持 `minimum`/`maxItems`/`uniqueItems` 等关键字，所以这两个 schema 不含它们，范围检查留在客户端（`GarfTool`/`propose_candidates` 与契约校验）。`tools.garf.TOOL_SPEC` 系列仍供 JSON-lines runner 使用，不直接交给模型。
- **候选簿**（`s2/candidates.py`）：每个 observation 一本。GARF 结果用观测到的锚点换算为 `world_T_goal`，按 `geometrically_eligible_seeds` 标记 `needs_motion_validation` / `rejected` / `unscreened`。候选 ID 为 `<observation>/call-<n>/seed-<s>`，同一 seed 换锚点或碎片集合再调一次就是新候选，不会覆盖旧记录。一次结果里任何候选有问题（seed 非整数、重复、碎片集合与请求不符），整批都不登记。给模型的摘要只含筛选结论和每片到目标的平移/转角，不含矩阵。
- **Claude 规划器**（`s2/claude.py`）：模型 `claude-opus-5-5`，`output_config.effort = high`，`tool_choice = auto`（Opus 5.5 拒绝强制 tool_choice），默认 `fallbacks: "default"`（beta `server-side-fallback-2026-07-01`，被安全分类器拒答时由服务端换模型重跑；trace 记录 `response.model` 与是否发生 fallback）。`tool_choice` 带 `disable_parallel_tool_use: true`，每轮最多一次工具调用；若仍收到多个调用，全部以错误结果回应、一个都不执行，避免模型引用它还没看到筛选结果的候选。手写循环：先查 `stop_reason`，`refusal` 与 `max_tokens` 的那一轮工具一概不执行；开启 fallback 时只执行最后一个 `fallback` 块之后的工具调用，回填历史时去掉其之前那一跳的 thinking/tool_use 块；是否由 fallback 模型作答以 `usage.iterations` 中的 `fallback_message` 为准；提交被拒时把原因作为 `is_error` 的 tool_result 回给模型；结束轮没有提交时提醒一次，再没有就返回 `no_request`；最多 `s2.max_turns` 轮。SDK 异常映射为带 `retryable` 的 `error`，非 API 异常照常抛出。密钥从环境或 `ant auth login` 读取，代码中没有密钥。尚未做过真实 API 调用。
- **离线规划器**：`ScriptedPlanner` 走同一套解析与校验，只在显式选择时使用，不会在模型不可用时自动顶替。
- **GARF 进程客户端**（`s2/garf_client.py`）：runner 在独立进程组中运行，超时时整组杀掉（包括 `uv run`/`sh -c` 这类包装），下次调用重启；首个请求用较长的启动时限。stdout 上的非 JSON 杂行被跳过并计数；JSON 回复若与请求的 seeds、observation 或碎片集合对不上，判为失步，杀掉 runner 并返回 `failed`。进程退出同样返回 `failed`，不抛异常。子进程环境去掉 `ANTHROPIC_*` 变量。缓存键 = 几何注册表内容哈希 + fragment IDs + anchor + seeds，只有缓存命中才把 observation ID 改写为当前值；只缓存所有 seed 都成功的结果；世界目标每个 observation 重新计算。
- **编排器**（`s2/orchestrator.py`）：每个规划事件都重新观察。交给执行器之前，编排器自己再做一遍校验：用当前 observation 跑 `validate_skill_request`，GARF 目标位姿必须与候选簿登记的 `world_T_goal` 逐位相同且该候选通过筛选，规划耗时不得超过 `s2.max_planning_s`。`reobserve` 不调用执行器。只有 place/insert、完成判据为 `released_stable`、执行完成**且**放手后的独立核验通过时才合并已验证簇，并要求簇里包含本次碎片、至少两片、全是已知碎片，否则记 `merge_rejected`。失败与空转计数只在一次成功执行后清零，所以“失败→reobserve→失败”也会触发熔断。某一步抛异常时，`run()` 返回 `error` 状态并保留之前的全部逐步记录。融合后的簇几何重新注册到 GARF runner 尚未实现。
- `python -m tasks.dual_arm_crag_skull.s2 preview --observation obs.json --trained-skills pick,place --with-garf` 打印将发送的完整首个请求（图片数据省略），不发送任何内容。关键帧应先缩到长边 ≤ 1568 px 再交给 S2。

## 基线公共层（已实现；基线尚未实跑）

所有方法（我们的 S2/S1/S0、DexAgent、ASPIRE）用同一份任务定义、同一个后端边界、同一个评分。代码在 `baselines/`，各基线的适配说明见 `baselines/README.md`。

- **任务**（`baselines/task.py`）：`AssemblyTask` 记录：
  - 碎片网格（layout 的 npz，`v`/`f`，米，碎片坐标系）及其 sha256；
  - 每片的起始位姿与目标位姿 `world_T_fragment`；
  - 锚点和 `goal_source`（`oracle_layout` 或 `garf_candidate`）；
  - 桌面高度与范围、装配台范围、机器人禁区；
  - 成功判据：容差和放手后的静置时间 `settle_s`；
  - 种子。

  `task_from_layout` 把 layout 的装配坐标系放在装配台中心的桌面上（不旋转），这是本模块显式规定的约定。锚点一开始就在目标位姿，并由夹具固定在那里，整个试验都不能动（协议项 `baselines.anchor = "fixture"`，`require_canonical` 会核对；锚点单放不稳）；其余碎片在 staging 位姿。它同时检查：
  - 网格文件必须在 layout 目录内；
  - 每片网格在目标位姿下都在装配台上方、不低于桌面；
  - 装配台和禁区都在桌内；
  - 摆放区足迹在桌内、不压装配台、互不重叠。

  默认任务 id 含 layout 哈希前缀。所有数组构造后只读；scipy 1.12 的 `Rotation.from_matrix` 不接受只读数组，转换前要先复制。
- **唯一的协议**：协议写在 `configs/jing.json`，包括：
  - 种子分段：practice 101–125、selection 126–130、held-out 1–100，暂定；
  - 初始条件参数；
  - 成功判据：`s2.completion_defaults.place`，即 `released_stable`，放手 1 s 后 3 mm / 3°，暂定，待打印件试配后替换；
  - 留出种子的占用目录 `baselines.heldout_claims_dir`。

  `require_canonical(config)` 拒绝其他配置。`verify_task(task)` 按协议从 layout 重建任务，容差、摆放、种子段标签、layout 或网格任何一处不同都会拒绝。两个运行器在每次试验前都会核对任务。
- **初始条件**：`initial_condition(task, seed, ...)` 只对 staging 碎片做平移（≤ `max_shift_m`）和绕竖直轴的偏航（≤ `max_yaw_deg`），并保证：
  - 足迹与其他碎片（含锚点）和禁区相距至少 `gap_m`；
  - 在桌内并留出同样的边距；
  - 不压装配台（允许贴边）。

  找不到摆法就报错，不会放宽约束。种子段一律由种子按标准分段推出。生成器是公开代码，留出集靠运行器的种子检查和占用记录保护，而不是靠保密。
- **评分**（`baselines/scoring.py`）：`score_assembly(task, final_world_T_fragment, settled=...)` 用仿真器结束后的真值位姿：
  - 按相对锚点的误差判断每片是否到位，锚点自己也要在容差内；
  - 缺失、非有限或非刚体的位姿都算失败，后两种记入 `invalid_ids`；
  - 不是在放手静置后读到的位姿，不算成功；
  - 目标来自 GARF 候选的任务直接拒绝评分，因为评分必须对照真值装配（同时携带两套目标尚未实现）；
  - 评分本身记下 layout 哈希、起始位姿摘要、目标来源和容差。
- **运行记录**：`RunRecord` 记录：
  - 方法、变体、任务；
  - 种子与种子段，种子段必须是该种子的标准分段；
  - 运行后端 `backend`，以及只用于验证流程的 `test_only` 标记；
  - 状态、评分、API 费用、模型调用次数、耗时和练习用的仿真试验次数。

  `summarize` 按方法、变体、任务、后端、`test_only` 和种子段分组，给出成功率及 Wilson 95% 区间。以下情况拒绝统计：
  - `test_only` 记录（除非显式要求）；
  - 同一留出种子记录两次；
  - 存储的 success 与评分矛盾；
  - 种子段不是标准分段；
  - 同组评分用了不同的 layout 或容差。
- **留出种子只用一次**：
  - `claim_heldout` 在试验动任何东西之前，在占用目录里以独占方式创建占用文件。换记录文件、换进程都无法再占同一个种子。
  - 试验中途崩溃或被中断时占用保留，运行器把这次记为 `error`。
  - `summarize` 列出没有对应记录的占用。
  - 新变体（例如改过的 ASPIRE 程序）可以再占同一种子，但每次占用都留在目录里可查。
- **后端**（`baselines/backend.py`）：`AssemblyBackend` 是仿真器边界，提供：
  - `reset(task)`；
  - `robot_state()`：时钟、关节和宽度必须有限，TCP 必须是刚体变换；
  - `observed_fragment_poses()`：每个位姿带来源和观测时刻 `time_s`；
  - `camera_frames()`：每帧带 `time_s`（曝光/采样时刻，与 `RobotSnapshot.time_s` 同一时钟，可以早于最新的机器人状态），内参必须是校正后零斜切的 3×3；
  - `step(...)`：一个控制周期的关节与夹爪宽度目标；
  - `privileged_fragment_poses()`：只用于评分。

  **每次试验的收尾对所有方法一样**：两爪张到最大，手臂保持 `settle_s`，期间照常经过命令检查，然后才读真值位姿评分。还夹在手里或松手后仍在滑动的碎片，按它停下的位置判定。

  对下发内容的检查（限位、碰撞）由调用方负责，不在后端里。Isaac 实现是 `sim/backend.py`：物理 240 Hz、目标 60 Hz（每周期 4 步线性插值）、相机 10 Hz 按需渲染（暂停物理，`World.play()` 恢复），锚点为运动学刚体，`reset` 先落定（最近 0.5 s 漂移 <0.1 mm / 0.05°，否则拒绝）再从 0 计时，所以试验从碎片落定处开始，不完全是任务里的初始位姿；`close()` 不关 Kit，进程最后调用 `shutdown(exit_code)`。`KinematicBackend` 只用于测试：没有物理，闭合的夹爪在 2 cm 内抓住最近的、没被另一只手夹着的碎片并刚性带走，张开即放下，碎片停在原处。它的结果永远不作为基线成绩。
- **直接下发关节的检查**（`s0/adapter.py`）：`check_joint_path(model, limits, *, now, q_now, qd_now, qd_window_s, times, joints)` 把一段关节目标视为在各自时刻到达：
  - 每个节点的加速度 = 相邻两段速度之差 / 两段时长的平均；第一个节点的「前一段」是实测速度的平均窗口 `qd_window_s`（`RobotSnapshot.velocity_window_s`，后端必须声明，非有限或为负即拒绝）。三个仿真后端报的都是上一控制周期的平均速度（Isaac 实测 2026-10-19：PhysX 速度就是刚执行完的指令段速度），窗口 = 一个周期，所以路径第一步与之后每一步同样读数，同一运动切成几段检查与整段检查读数相同；瞬时速度（窗口 0）则第一段按从 `now` 起匀加速读成 `2 (v1 − qd_now) / dt`。加速度上限不变；新读法从不比旧读法（一律 `2 (v1 − qd_now) / dt`）更严，历史记录里没有被它改变的判定（见 TODO）；
  - 最后一个目标之后必须能在关节限位内制动停下；
  - 再用 `check_geometry` 的臂间、桌面、TCP 高度规则检查。

  实测状态或时钟不是有限值时直接拒绝。不裁剪也不重新定时，只拒绝。ASPIRE 适配器的每一条关节命令都先过这道检查。
- **ASPIRE 适配**（`baselines/aspire/`，说明见该目录 README）：
  - 坐标：ASPIRE 的基座系 `robot_base` 就是任务世界系（米，z 向上），四元数 wxyz；相机位姿是 `robot_base_T_camera`，OpenCV 轴，图像时间戳取 `CameraFrame.time_s`。
  - 手臂名用 `left`/`right`，ASPIRE 默认的 `primary`/`secondary` 经 `resolve_arm` 映射为 left/right。夹爪 0 是 S0 配置宽度范围的最小值（闭合），1 是最大值，按宽度线性换算。
  - 程序能拿到的任务信息都在 `get_task_context().metadata`：碎片 id、锚点、目标位姿及其来源、碎片系包围盒、容差、桌面高度、控制器限位说明，以及每次调用都刷新的观测位姿和来源。程序拿不到种子、种子段和特权位姿。
  - 每条关节命令（单步、轨迹、`go_home`）整段先过 `check_joint_path`，周期之间再按 `servo_hz` 插值补查几何；轨迹末尾加一个保持周期，保证指令运动以静止结束。被拒的命令什么都不动，返回 `safety_interlock`；被拒的夹爪命令在适配器里直接抛出，因为 CapApi 会把失败的夹爪目标带进后续运动。不裁剪，不重新定时。
  - `action_bounds` 取 0.9·min(vmax, amax·dt/2)：ASPIRE 自带的关节插值在一个周期内起停，从静止开始一个周期达到速度 v 需要 2v/dt 的匀加速度，这样它的轨迹也满足 S0 的加速度上限。
  - 贴桌接触：程序调用扩展工具 `fanuc_crx.declare_contact(kind, fragment_id, arm=...)`（`grasp` 或 `place`）和 `fanuc_crx.clear_contact(arm=...)`。程序只给碎片名，窗口中心由适配器取：抓取用碎片的观测位姿，放置用它的目标位姿，符合上面「不能用程序自己给的坐标」的约定。手臂已在接触带内时不许改窗口，要先抬出来。有窗口时 `action_bounds` 再按手指角点不超过 S0 接触速度的 80% 收紧。所有方法用同一套硬件：CRX + YAM，后端为 60 Hz 的运动学替身或 Isaac。
  - `check_success` 先按上面的统一收尾（两爪张开、保持 1 s，经 S0 检查，不占程序的步数预算），再评分；S0 拒绝这次保持时（程序结束时手臂还在快速运动），评分不算静置，也不算成功。
  - 种子守卫：practice 模式只跑 practice 和 selection 种子，evaluation 模式只跑 held-out 种子；种子分段必须与 `jing.json` 一致；held-out trial 必须写入记录，同一程序变体在同一任务的同一种子上只能跑一次。
- **DexAgent 适配**（`baselines/dexagent/`，说明见该目录 README）：
  - 坐标：`rig_T_world` 由偏航（默认 0）和平移组成，把装配台中心放在 rig (0.00, 0.40) 处，桌面为 rig z = 0。物体坐标系就是碎片坐标系，评分用 `world_T_fragment = inverse(rig_T_world) @ rig_T_body`；固定的锚点保持记录的起始位姿。
  - 物体 id 按碎片 id 的顺序命名为 `<颜色>_fragment`，根刚体为 `<id>_<id>`，自由关节为 `<id>_<id>_free`。对照表、`rig_T_world`、任务记录和所有文件的 sha256 都写在 `inputs/export_manifest.json`。
  - 模型中转：DexAgent 以 `provider: external` 把每个请求写成文件，由我们的中转调用 Claude 应答。中转保留 API 返回的完整助手内容（含 thinking 与签名）原样回传，并核对 DexAgent 的历史与已应答的内容一致；被拒、截断或停止原因不是 `end_turn`/`tool_use` 的轮次，其工具调用一律不转交；每次调用前按最坏情况检查美元上限；停止时写 `operator_checkpoint.json` 并回复 `report_blocked`。
  - 状态映射到 `RunRecord.status`：中转因预算停止 → `budget_exhausted`，因拒答停止 → `rejected`，其他停止 → `error`；我们的墙钟截止 → `timeout`；DexAgent 自己报 blocked 或停滞 → `blocked`；交付通过复核 → `completed`。
  - 留出种子要显式允许、必须写入记录，同一变体在同一任务的同一种子上只跑一次（`scoring.refuse_repeats`，ASPIRE 同样适用）。
  - 共享硬件移植（`--bench crx_yam`，变体 `crx_yam_stage3`，进行中）：rig 是世界坐标下移到桌面（不偏航、不平移）。Stage 3 的门禁里有 `s0_delivery_check`，DexAgent 只接受 S0 在完美跟踪下不会拒绝的交付。交付由 `executor.py` 在 `AssemblyBackend` 上执行：参考指令从记录的起始状态按 60 Hz 斜坡，分块经 `check_joint_path` 检查。手臂指令每次进入接触带的那个采样点（或前一个采样点，取指令速度变化更小者）都另起一块，窗口在这时按该臂第一个带内指令的阶段标签、用此刻的观测声明（抓取用碎片的观测位姿；放置时碎片在目标的 S0 半径内用目标位姿，否则用碎片位姿；锚点由夹具固定，永远不是接触对象），出带即清除；其余各块不超过 1 s，在指令速度变化最小处切开（S0 曾把块的第一段读成 2Δv/dt；统一口径后不再必需，保留无害）。被 S0 拒绝的块不改变任何窗口。无论完成还是被拒，都走统一收尾（两爪张开、保持 `settle_s` 经 S0 检查、评分）。后端落定的起点与 DexAgent 规划的起点相差超过 1 mm / 1° 时不执行；每个交付只执行一次（开动前建 `execution.started`，执行中出错记为 `error`；留出种子必须写记录）。状态：执行完 → `completed`，S0 拒绝 → `rejected`，不执行 → `error`。
