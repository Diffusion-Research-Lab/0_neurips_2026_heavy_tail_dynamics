"""Command-line entrypoint for datakit."""

from ._prepare import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli())
