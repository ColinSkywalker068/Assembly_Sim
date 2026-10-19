import pytest

from tasks.dual_arm_crag_skull.baselines.scoring import CLAIMS_ENV


@pytest.fixture(autouse=True)
def isolated_heldout_claims(tmp_path_factory, monkeypatch):
    """Every test claims held-out seeds in its own temporary directory, never in the canonical one."""
    monkeypatch.setenv(CLAIMS_ENV, str(tmp_path_factory.mktemp('claims')))
