"""Capsule CLI - product-grade interface for generating and verifying receipts.

Commands:
    emit    - Generate a portable verification artifact (.cap file)
    verify  - Verify a capsule with stable exit codes for CI
    inspect - Display capsule metadata
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from .emit import emit_command
from .verify import verify_command
from .inspect_cmd import inspect_command


@click.group()
@click.version_option(version="0.1.0", prog_name="capsule")
def cli() -> None:
    """Capsule CLI - cryptographic receipt generation and verification.

    Generate portable verification artifacts from traces and verify them
    with stable exit codes suitable for CI integration.
    """


cli.add_command(emit_command, name="emit")
cli.add_command(verify_command, name="verify")
cli.add_command(inspect_command, name="inspect")


def main() -> None:
    """CLI entry point."""
    cli(prog_name="capsule")


if __name__ == "__main__":
    main()
