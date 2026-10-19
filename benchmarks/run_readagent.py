"""Standalone ReadAgent experiment: python -m benchmarks.run_readagent."""
from benchmarks.run_single import main as run_single


def main(argv=None):
    return run_single("readagent", argv)


if __name__ == "__main__":
    main()
