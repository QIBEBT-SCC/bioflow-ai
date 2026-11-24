"""CLI wrapper for upfront network and proxy connectivity checks."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

from bioflow_ai.cli.initialization import tui
from bioflow_ai.services.initialization.network import run_network_preflight

if TYPE_CHECKING:
    from bioflow_ai.core.config import NetworkConfig


def verify_network_connectivity(network: NetworkConfig) -> bool:
    """Verify proxy and external registry/Hugging Face reachability before continuing."""
    tui.render_section(
        'Network Connectivity Check',
        'Verify proxy connectivity and access to required container registries and Hugging Face.',
    )

    with tui.console.status('Checking network connectivity...') if tui.console.is_terminal else nullcontext():
        result = run_network_preflight(network)

    tui.render_network_preflight_results(result)
    if result.ok:
        tui.render_notice('All required network checks passed.', style='#10b981')
        return True
    return False
