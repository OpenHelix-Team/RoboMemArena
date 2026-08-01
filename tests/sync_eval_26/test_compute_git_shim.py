from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO_ROOT
    / "evaluation_benchmark"
    / "sync_eval_26"
    / "scripts"
    / "run_exact20_dual_gpu.sh"
)


def test_runner_requires_a_real_git_binary_for_locked_runtime_verification() -> None:
    content = RUNNER.read_text(encoding="utf-8")

    assert "command -v git" in content
    assert "git is required to verify the locked official runtime" in content
    assert "runtime_tools" not in content
