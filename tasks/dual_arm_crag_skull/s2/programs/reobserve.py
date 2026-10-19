"""Minimal frozen skill-mode program; plans a fresh observation, never moves."""
observation = get_observation()
result = submit_skill_request(
    observation_id=observation['observation_id'],
    skill_id='reobserve',
    active_arms=[],
    rationale='Acquire a fresh observation before choosing a trained manipulation skill.',
)
