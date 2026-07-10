import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_imports_without_qwen_dependencies():
    import_script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def reject_optional_dependencies(name, *args, **kwargs):
            if name == "transformers" or name.startswith("transformers."):
                raise AssertionError(f"unexpected optional dependency import: {name}")
            if name == "triton" or name.startswith("triton."):
                raise AssertionError(f"unexpected optional dependency import: {name}")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = reject_optional_dependencies

        import research.kmd2_ablation as suite

        assert suite.SUITE_VERSION == "1.0.0"
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", import_script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
