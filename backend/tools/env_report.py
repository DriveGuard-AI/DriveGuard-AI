"""Print the interpreter and package versions this environment actually runs.

The application backend, the live inference service, the perception worker and
the detector service each use their own interpreter. This script reports one of
them as JSON so the four can be compared without sharing code between
repositories.

    python tools/env_report.py            # human readable JSON
    python tools/env_report.py --names    # only the package names it checks
"""
import argparse
import json
import platform
import sys
from importlib import metadata

PACKAGES = (
    "fastapi", "uvicorn", "sqlalchemy", "alembic", "pydantic",
    "torch", "torchvision", "numpy", "opencv-python", "mediapipe", "ultralytics",
)


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "python_tag": "{}.{}".format(*sys.version_info[:2]),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", action="store_true", help="list the checked packages and exit")
    args = parser.parse_args(argv)
    if args.names:
        print("\n".join(PACKAGES))
        return
    print(json.dumps(environment(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
