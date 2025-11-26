from __future__ import annotations

import typer
from dotenv import load_dotenv

from xray.bragg.cli import bragg_cli
from xray.extension.cli import extension_cli

# Load environment variables from .env file
load_dotenv()

app = typer.Typer()

app.add_typer(bragg_cli, name="bragg")
app.add_typer(extension_cli, name="extension")


def main() -> None:
    """Entry point that runs the Typer-powered CLI."""
    app()


if __name__ == "__main__":
    main()
