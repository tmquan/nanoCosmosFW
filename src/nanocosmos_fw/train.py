"""Entry point that registers the EM recipe before Cosmos Framework starts."""

from nanocosmos_fw.experiment import register_experiment


def main() -> int:
    register_experiment()
    from cosmos_framework.scripts.train import main as cosmos_main

    return cosmos_main()


if __name__ == "__main__":
    raise SystemExit(main())
