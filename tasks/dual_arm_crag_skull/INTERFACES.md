# 分层接口与数据约定（设计，尚未实现执行器）

## 坐标与时间

米、秒、弧度；世界 Z 向上。`T_A_B` 将 B 坐标变换到 A。原始 pose 用 `[x,y,z,qw,qx,qy,qz]`，存储单位四元数。记录 world/base_left/base_right/flange/TCP/camera 外参及版本。相机光学轴转换显式保存，不能混用 USD 与 OpenCV 轴。

所有流保存采样时间、接收时间和时钟域，转换到 episode 单调时间轴。同步图像按曝光/采样时刻匹配实际状态；不得将执行后状态与刚执行动作直接当成下一步监督对。

## S2 -> S1

SkillRequest 包含 request_id、skill_id、fragment_id、active_arms、target_frame、target_pose、support_fragment_id、completion_predicate、timeout_s。技能候选为 pick/place/align/insert/support/retreat/reobserve；模型只能选择已经训练和实现的技能。target 必须来自可观测/预测几何或明确给定条件，记录 provenance。初期可人工提供请求。

## S1 -> S0

ActionChunk 包含 request_id、chunk_id、observation_time、valid_until、左右 anchor 的 T_world_tcp、严格递增的 waypoint_times、20D waypoint 数组、控制模式。

数组顺序固定为 left(position3, rotation6, width1), right(position3, rotation6, width1)。每手每个 waypoint 表示 `T_anchor_target = inverse(T_world_anchor) @ T_world_target`。anchor 在整块预测期间保持固定；宽度为绝对米值。S0 用 `T_world_anchor @ T_anchor_target` 还原绝对目标。

rotation6 先放旋转矩阵第一列三个数，再放第二列三个数。解码：b1=normalize(a1), b2=normalize(a2-dot(b1,a2)b1), b3=cross(b1,b2)。拒绝非有限、零长度或近共线输入，不能默默生成任意姿态。非活动臂也必须有明确 hold 目标。

S0 返回 accepted/rejected/holding/completed/failed、reason、measured_tcp、measured_width、tracking_error、timestamp。高层失败与动作完成不等于装配成功；装配由独立 evaluator 判断。碰撞检测包含 arm-arm、arm-object、tool-object、table；预期接触使用明确接触对象/阶段，不关闭全场碰撞。

## Episode

建议三种 source：sim_expert / umi / robot_teleop。共同字段：

- episode_id、task_instruction、skill_segments、fragment_ids、geometry_hash、calibration_version、seed、source。
- images.left_wrist/right_wrist/agent 与各自 timestamps、可用性 mask。
- state.tcp_pose_left/right、gripper_width_left/right、inter_tcp_pose。
- action.requested_tcp/width、accepted_tcp/width、issue_time、scheduled_time；UMI 示范为未来实际 TCP 轨迹，另记 action_origin，不冒充机器人命令。
- diagnostics.joint_positions/velocities（UMI 缺失）、tracking、接触/力（若有）。
- privileged.gt_object_poses（仅仿真专家/评估）、末态误差、success、failure_reason。

先保存高频原始数据，再按 10 Hz 策略时间网格插值状态和对齐图像，旋转用 SLERP。保留完整 episode，不跨 episode/技能边界错误拼接动作块。训练可分别评估 command-target 与 achieved-trajectory 监督，不能未经标记混合二者。

S1 默认不读取 privileged 或 joint diagnostics，确保仿真/UMI 输入一致。训练集统计独立保存并复用于推理。按 episode/初始条件划分集合，不能随机切相邻帧造成泄漏。

## 观测构建

S1 输入三视角历史 + 双 TCP/宽度 + 双手相对位姿 + 技能条件/目标；S2 使用关键帧、可观测场景图、CRAG 预测和反馈。第一版原始 DP 没有现成的语言大模型接口，采用明确 skill ID 与目标编码；语言到技能的映射由 S2 完成。

UMI 外部相机缺失时显式标记，需训练对应 missing-view 模式；不生成假的同步第三视角。相机定位误差、夹爪几何差异和真实延迟属于 sim-to-real 验证项。

## RGB-D、完整 part 点云与 GARF

完整扫描点云定义每片的局部几何 `P_i`，需统一到米并提供一致定向法向量。没有法向量时必须估计并验证，不能填零。只有 XYZ 的点云可以先作为定位模板，但尚不满足此 GARF adapter 输入契约。

RGB 与 depth 必须已对齐、去畸变。对实例掩码内像素，光学 Z 深度反投影：

`p_camera = [(u-cx) z/fx, (v-cy) z/fy, z]`

`p_world = world_T_camera @ [p_camera, 1]`

腕相机外参随关节运动：`world_T_camera(t) = world_T_base @ base_T_flange(q(t)) @ flange_T_camera`；三路图像及关节状态必须按采样时间对齐。USD 相机轴转 OpenCV 光学轴使用 `diag(1,-1,-1,1)`，深度使用 `distance_to_image_plane`，不能把沿射线的距离当 Z。

实例分割/身份匹配 → 全局粗配准 → 局部 ICP → 重叠/误差/对称性检查 → `world_T_part_current`。已实现 `depth_to_world` 和需要初始位姿的 `refine_pose`；没有实现自动分割、全局初始化或身份匹配。ICP 残差低不代表身份正确，对称碎片尤其需要多视角验证。

GARF 输入完整局部几何，输出 `assembly_T_part_goal`。锚点观测恢复 `world_T_assembly` 后得到目标；当前位姿与目标之差用于规划。S1 可以继续吃 RGB+机器人状态，把 depth 用于几何工具；若 S1 直接吃 RGB-D，需要相应编码器与训练，不能把深度直接加到既有 RGB checkpoint。

## 多噪声与逐步重装配

`tools.garf.TOOL_SPEC` 与 `CANDIDATE_TOOL_SPEC` 可作为 Anthropic tools 定义。后者接受 `seeds`（1–8 个不重复整数），点采样固定 seed=42，噪声随 seed 改变；结果保留所有候选，无默认赢家。生产筛选必须基于观察与几何，离线 `evaluate.py` 才能读取 GT。

已稳定装配集合 C 的输入可以合并为一个刚体：

`P_C = union_i (cluster_T_i @ P_i)`

`assembly_T_i = assembly_T_C @ cluster_T_i`

`fuse_verified_cluster` 要求外部独立确认稳定性，不能用 GARF 的预测充当“已经装好”。工具返回成员映射，S2 保留原始 fragment ID；`expand_cluster_pose` 恢复每片预测。若接触松动、位姿漂移或遮挡导致不确定，应拆回独立状态并重新观察。当前融合保留内部接触面，属于分布偏移实验；表面去重/遮蔽及真实闭环融合尚未验证。

建议状态机：observe → sample_candidates → validate/rank → plan_skill → execute → release_and_verify → merge_verified_cluster → reobserve。每次只执行已验证的下一步；场景变化使旧候选失效。相同已知几何的候选可缓存，但锚点位姿和可达性必须重新计算。超时/取消由调用进程管理；当前 JSON-lines runner 尚无网络服务或内部抢占式超时。
