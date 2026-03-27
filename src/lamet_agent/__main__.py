"""Allow `python -m lamet_agent` to behave like the installed CLI."""

from lamet_agent.cli import entrypoint


if __name__ == "__main__":
    entrypoint()
