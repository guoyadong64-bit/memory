"""Standalone A-Mem experiment: python -m benchmarks.run_amem."""
from benchmarks.run_single import main as run_single


def main(argv=None):
    return run_single("amem", argv)


if __name__ == "__main__":
    main()
