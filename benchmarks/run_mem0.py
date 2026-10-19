"""Standalone Mem0 experiment: python -m benchmarks.run_mem0."""
from benchmarks.run_single import main as run_single


def main(argv=None):
    return run_single("mem0", argv)


if __name__ == "__main__":
    main()
