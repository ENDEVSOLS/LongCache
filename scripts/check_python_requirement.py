"""Verify that a real Python 3.10 install rejects the built v0.2 wheel."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--wheel", required=True, type=Path)
    args = parser.parse_args()
    assert args.wheel.is_file(), "Wheel not found"
    version = subprocess.check_output(
        [args.python, "-c", "import sys; print('.'.join(map(str,sys.version_info[:2])))"], text=True
    ).strip()
    assert version == "3.10", "Negative install must use a real Python 3.10 interpreter"
    with tempfile.TemporaryDirectory() as target:
        result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                args.python,
                "--no-deps",
                "--target",
                target,
                str(args.wheel.resolve()),
            ],
            capture_output=True,
            text=True,
        )
    message = (result.stdout + result.stderr).lower()
    assert result.returncode != 0, "Python 3.10 unexpectedly accepted the wheel"
    assert "python" in message and "3.11" in message, message
    print("PASS: real Python 3.10 rejects the wheel's Requires-Python >=3.11")


if __name__ == "__main__":
    main()
