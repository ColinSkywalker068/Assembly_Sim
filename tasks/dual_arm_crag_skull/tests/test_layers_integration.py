"""S2 -> S1 -> S0 plumbing with the real CRX kinematics: a scripted S2, a stand-in S1, the real S0."""
import numpy as np

from tasks.dual_arm_crag_skull.contracts.actions import ARMS, encode_chunk
from tasks.dual_arm_crag_skull.s0.adapter import S0Adapter, S0Limits
from tasks.dual_arm_crag_skull.s0.kinematics import ROBOTIQ_PROXY_MAX_WIDTH_M, load_kinematics, robotiq_proxy_model
from tasks.dual_arm_crag_skull.s2.orchestrator import AssemblyOrchestrator, ExecutionReport, Verification
from tasks.dual_arm_crag_skull.s2.planner import ScriptedPlanner
from tasks.dual_arm_crag_skull.tests.s2_fakes import COMPLETIONS, TIMEOUTS, observation

MODEL = robotiq_proxy_model()
HOME = np.array([0.42959, -0.22024, -0.68657, -0.00021, -1.10447, -0.42928])  # reachable, away from limits
LIMITS = S0Limits(max_joint_velocity_rad_s=load_kinematics()['max_joint_velocity_rad_s'],
                  max_joint_acceleration_rad_s2=np.full(6, 20.0),  # test-only: real CRX value unknown
                  gripper_width_range_m={arm: (0.0, ROBOTIQ_PROXY_MAX_WIDTH_M) for arm in ARMS},
                  arm_arm_clearance_m=0.02, table_top_z_m=0.75, table_clearance_m=0.01, tcp_min_height_m=0.02,
                  servo_hz=60, max_observation_age_s=0.5, anchor_position_tolerance_m=0.002,
                  anchor_rotation_tolerance_rad=0.005)


class StandInS1:
    """Moves the requested arm 1 cm per 0.1 s straight down and closes its gripper; others hold."""

    def __init__(self, lag_s=0.0):
        self.lag_s = lag_s

    def chunk(self, request, observation_time, joints, widths):
        anchors = {arm: MODEL.arms[arm].tcp(joints[arm]) for arm in ARMS}
        targets = {arm: np.repeat(anchors[arm][None], 4, axis=0) for arm in ARMS}
        chunk_widths = {arm: np.full(4, widths[arm]) for arm in ARMS}
        for arm in request.active_arms:
            targets[arm][:, 2, 3] -= 0.01 * np.arange(1, 5)
            chunk_widths[arm] = np.linspace(widths[arm], 0.03, 4)
        return encode_chunk(request_id=request.request_id, chunk_id=0, observation_time=observation_time,
                            valid_until=observation_time + 0.15, anchors=anchors,
                            waypoint_times=observation_time + 0.1 * np.arange(1, 5), targets=targets,
                            widths=chunk_widths, active_arms=request.active_arms)


class Executor:
    def __init__(self, s1):
        self.s1, self.adapter, self.log = s1, S0Adapter(MODEL, LIMITS), []

    def execute(self, request, obs):
        joints, widths = {arm: HOME.copy() for arm in ARMS}, {'left': 0.08, 'right': 0.05}
        chunk = self.s1.chunk(request, obs.time, joints, widths)
        # The robot has not moved since the observation, so the TCP it had then is the TCP at HOME.
        measured_then = {arm: MODEL.arms[arm].tcp(joints[arm]) for arm in ARMS}
        # The grippers sit where they were last commanded, so measured and commanded widths agree here.
        status, trajectory = self.adapter.plan(chunk, now=obs.time + 0.02 + self.s1.lag_s, joints=joints,
                                               joint_velocities={arm: np.zeros(6) for arm in ARMS}, widths=widths,
                                               commanded_widths=dict(widths), tcp_at_observation=measured_then)
        self.log.append((chunk, status, trajectory))
        if status.status != 'accepted':
            return ExecutionReport('rejected', reason=status.reason)
        return ExecutionReport('completed', details={'servo_samples': len(trajectory.times)})


class Observer:
    def __init__(self):
        self.count = 0

    def observe(self, clusters):
        self.count += 1
        return observation(f'obs-{self.count}', clusters=clusters)


class PassingVerifier:
    def verify(self, request, before, after):
        return Verification(True, reason='grasp detected')


def pick(context):
    return {'observation_id': context.observation.observation_id, 'skill_id': 'pick', 'active_arms': ['left'],
            'fragment_id': 'part_1', 'support_fragment_id': None, 'target_candidate_id': None,
            'rationale': 'part_1 is localized and free'}


def run(executor):
    return AssemblyOrchestrator(observer=Observer(), planner=ScriptedPlanner([pick]), executor=executor,
                                verifier=PassingVerifier(), trained_skills=('pick',), completion_defaults=COMPLETIONS,
                                timeouts_s=TIMEOUTS, max_planning_s=60.0).run(1)


def test_request_id_flows_from_s2_through_chunk_to_s0_trajectory():
    executor = Executor(StandInS1())
    report = run(executor)
    chunk, status, trajectory = executor.log[0]
    assert report.steps[0]['execution']['status'] == 'completed'
    assert status.status == 'accepted' and status.request_id == chunk.request_id == report.steps[0]['skill_request']['request_id']
    assert np.allclose(np.diff(trajectory.times)[:-1], 1 / 60) and trajectory.times[-1] == chunk.waypoint_times[-1]
    np.testing.assert_allclose(trajectory.joints['right'], np.repeat(HOME[None], len(trajectory.times), axis=0))
    tcp = MODEL.arms['left'].tcp(trajectory.joints['left'][-1])
    np.testing.assert_allclose(tcp[:3, 3], MODEL.arms['left'].tcp(HOME)[:3, 3] - [0, 0, 0.04], atol=2e-4)
    assert trajectory.widths['left'][-1] == np.float64(0.03)


def test_late_s1_output_is_rejected_by_s0_and_counted_as_failure():
    executor = Executor(StandInS1(lag_s=0.2))
    report = run(executor)
    _, status, trajectory = executor.log[0]
    assert status.status == 'rejected' and 'stale' in status.reason
    assert trajectory.hold and trajectory.widths['left'][0] == 0.08  # braking from rest = hold in place
    assert report.steps[0]['execution']['status'] == 'rejected' and report.verified_clusters == ()
