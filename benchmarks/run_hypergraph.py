"""Standalone hypergraph memory experiment: python -m benchmarks.run_hypergraph."""
from benchmarks.run_single import main as run_single


def main(argv=None):
    return run_single("hypergraph", argv)


if __name__ == "__main__":
    main()
