"""R5: retrieval-gated MCP tool disclosure.

The bar R5 has to clear is stated in tokens, so these tests assert tokens, not
characters. The gate's whole point is to stop charging every session for a schema
surface most sessions never use -- while never making a tool unreachable, which
is the property that lets the default change safely.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp", reason="mcp not installed")

from archex.integrations.mcp import (
    ALL_TOOL_NAMES,
    DISCLOSURE_CORE_TOOL_NAMES,
    TOOL_SCOPE_PROFILES,
    DisclosureGate,
    build_server,
    measure_tool_schema_size,
    resolve_tool_scope,
)

#: R5's acceptance bar.
DISCLOSURE_TOKEN_BUDGET = 1000


class TestDisclosureBudget:
    def test_the_disclosure_scope_fits_the_acceptance_budget(self) -> None:
        report = measure_tool_schema_size(DISCLOSURE_CORE_TOOL_NAMES)
        assert report["total_tokens"] < DISCLOSURE_TOKEN_BUDGET

    def test_the_measured_cost_is_pinned_not_merely_under_the_bar(self) -> None:
        """`< 1000` tolerates 999, a 30% erosion of a 765-token result.

        The reported figure is what the decision document and CHANGELOG quote, so
        it is pinned directly. Widen deliberately if a core tool's schema changes.
        """
        report = measure_tool_schema_size(DISCLOSURE_CORE_TOOL_NAMES)
        assert 740 <= report["total_tokens"] <= 790, "measured 765 at R5"

    def test_disclosure_is_a_large_cut_against_the_full_surface(self) -> None:
        """Pins the published reduction, not a ratio the budget already implies.

        `disclosure / full < 0.30` could never fail independently: 0.30 of the
        full surface is above the budget test's own ceiling, and adding any tool
        outside the core *raises* the denominator, making it easier to pass. The
        published claim is the reduction, so assert that.
        """
        disclosure = measure_tool_schema_size(DISCLOSURE_CORE_TOOL_NAMES)["total_tokens"]
        full = measure_tool_schema_size(None)["total_tokens"]
        assert isinstance(disclosure, int)
        assert isinstance(full, int)
        assert 1 - disclosure / full > 0.75, "CHANGELOG and DECISION.md claim 80.2%"

    def test_the_measurement_reports_tokens_as_well_as_characters(self) -> None:
        """The bar is in tokens; measuring only chars left it uncheckable."""
        report = measure_tool_schema_size(None)
        assert report["total_tokens"] > 0
        assert report["total_chars"] > report["total_tokens"]
        assert set(report["per_tool_tokens"]) == set(report["per_tool_chars"])

    def test_per_tool_tokens_sum_to_the_total(self) -> None:
        report = measure_tool_schema_size(None)
        assert sum(report["per_tool_tokens"].values()) == report["total_tokens"]

    def test_a_single_tool_measures_to_a_known_value(self) -> None:
        """The sum-to-total check cannot detect a tokenization error; this can."""
        report = measure_tool_schema_size(frozenset({"context"}))
        assert 514 <= report["total_tokens"] <= 554, "measured 534 at R5"

    def test_every_disclosure_tool_is_a_registered_tool(self) -> None:
        assert frozenset(ALL_TOOL_NAMES) >= DISCLOSURE_CORE_TOOL_NAMES

    def test_the_core_membership_is_pinned_exactly(self) -> None:
        """The constant IS the contract, so equality, not a superset.

        A superset check tolerates shrinking it. Dropping `query_repo` would take
        archex's headline retrieval tool out of the first `list_tools()` and leave
        the suite green.
        """
        assert frozenset({"context", "query_repo"}) == DISCLOSURE_CORE_TOOL_NAMES

    def test_disclosure_is_selectable_as_a_named_scope(self) -> None:
        assert resolve_tool_scope("disclosure") == DISCLOSURE_CORE_TOOL_NAMES
        assert TOOL_SCOPE_PROFILES["disclosure"] == DISCLOSURE_CORE_TOOL_NAMES


class TestDisclosureGate:
    def test_a_disabled_gate_is_open_from_the_start(self) -> None:
        gate = DisclosureGate(enabled=False)
        assert gate.is_open is True
        assert gate.visible(None) is None

    def test_an_enabled_gate_starts_closed_and_hides_the_wider_surface(self) -> None:
        gate = DisclosureGate(enabled=True)
        assert gate.is_open is False
        assert gate.visible(None) == DISCLOSURE_CORE_TOOL_NAMES

    def test_retrieving_opens_the_gate_exactly_once(self) -> None:
        gate = DisclosureGate(enabled=True)
        first = next(iter(sorted(DISCLOSURE_CORE_TOOL_NAMES)))
        assert gate.observe_call(first) is True
        assert gate.observe_call(first) is False, "opening must be reported once, not per call"
        assert gate.is_open is True
        assert gate.visible(None) is None

    def test_a_non_retrieval_call_does_not_open_the_gate(self) -> None:
        """Otherwise any tool call would defeat the gate on the first turn."""
        gate = DisclosureGate(enabled=True)
        assert gate.observe_call("get_file_tree") is False
        assert gate.is_open is False

    def test_an_explicit_scope_still_bounds_what_the_gate_reveals(self) -> None:
        """`--tools core --disclosure` must never advertise outside `core`."""
        core = TOOL_SCOPE_PROFILES["core"]
        gate = DisclosureGate(enabled=True)
        gated = gate.visible(core)
        assert gated is not None
        assert gated < core, "a closed gate must narrow an explicit scope"
        gate.observe_call("context")
        assert gate.visible(core) == core

    def test_a_disabled_gate_never_opens_on_a_call(self) -> None:
        """`observe_call` must not claim a transition on an ungated server."""
        assert DisclosureGate(enabled=False).observe_call("context") is False

    def test_a_scope_disjoint_from_the_core_is_not_gated_to_nothing(self) -> None:
        """`graph` shares no tool with the retrieval core.

        Intersecting would advertise zero tools while every tool stayed callable,
        which is strictly worse than serving the scope. (The gate would still
        open -- an unadvertised `context` call dispatches fine -- so "it could
        never open" is not the reason.) A client that narrowed its own scope has
        already made the cost decision.
        """
        graph = TOOL_SCOPE_PROFILES["graph"]
        gate = DisclosureGate(enabled=True)
        assert gate.visible(graph) == graph

    def test_the_gate_reports_whether_it_is_enabled(self) -> None:
        assert DisclosureGate(enabled=True).enabled is True
        assert DisclosureGate(enabled=False).enabled is False


def test_an_unscoped_server_is_still_gated() -> None:
    """`--tools all` bounds what opens, it does not disable the gate.

    Documented explicitly because the opposite is the natural guess, and an
    operator who guessed wrong would think they had restored the full surface.
    """
    gate = DisclosureGate(enabled=True)
    assert gate.visible(resolve_tool_scope("all")) == DISCLOSURE_CORE_TOOL_NAMES


def test_disabling_disclosure_is_the_only_way_back_to_everything() -> None:
    assert DisclosureGate(enabled=False).visible(resolve_tool_scope("all")) is None


class TestListChangedCapability:
    """The gate is only safe if the client knows the list can change.

    A client honours `notifications/tools/list_changed` only when the server
    declared the capability at initialization, and the SDK defaults it off. If
    this regresses, the gate silently becomes a permanent tool-hiding mechanism
    for every spec-compliant client: it is entitled to never re-fetch, so the
    17 tools disclosed after the first retrieval would never become visible.
    """

    @staticmethod
    async def _declared_tools_changed(*, disclosure: bool) -> bool:
        from contextlib import asynccontextmanager
        from unittest.mock import patch

        from archex.integrations.mcp import run_stdio_server

        captured: dict[str, Any] = {}

        @asynccontextmanager
        async def fake_stdio_server() -> AsyncIterator[tuple[None, None]]:
            yield (None, None)

        class FakeServer:
            def create_initialization_options(self, notification_options: Any = None) -> object:
                captured["opts"] = notification_options
                return object()

            async def run(self, *args: object, **kwargs: object) -> None:
                return None

        with (
            patch("archex.integrations.mcp.build_server", return_value=FakeServer()),
            patch("mcp.server.stdio.stdio_server", fake_stdio_server),
            patch("archex.serve.runtime.QueryRuntime.close"),
        ):
            await run_stdio_server(disclosure=disclosure)

        opts = captured["opts"]
        assert opts is not None, "no notification options were declared at all"
        return bool(opts.tools_changed)

    @pytest.mark.asyncio
    async def test_the_gated_server_declares_the_capability(self) -> None:
        assert await self._declared_tools_changed(disclosure=True) is True

    @pytest.mark.asyncio
    async def test_the_ungated_server_does_not_promise_changes(self) -> None:
        """An ungated tool list genuinely never changes; saying it might is a lie."""
        assert await self._declared_tools_changed(disclosure=False) is False

    def test_the_capability_reaches_the_wire_shape(self) -> None:
        """Guards the SDK contract the above relies on: the flag really does
        become `listChanged` in the declared tools capability."""
        from mcp.server.lowlevel import NotificationOptions

        server = build_server(disclosure=True)
        opts = server.create_initialization_options(
            notification_options=NotificationOptions(tools_changed=True)
        )
        assert opts.capabilities.tools is not None
        assert opts.capabilities.tools.listChanged is True


class TestGateThroughARealSession:
    """Drives the gate through `build_server`'s own handlers over a real client
    session, because the premise the whole change rests on -- an unadvertised
    tool still dispatches -- had no coverage at all.

    `_run_mcp_tool` is stubbed so these test the gate rather than the tools: no
    index, no repo, deterministic.
    """

    @staticmethod
    @asynccontextmanager
    async def _session(
        *, disclosure: bool, tool_names: frozenset[str] | None = None
    ) -> AsyncIterator[tuple[Any, list[Any]]]:
        from unittest.mock import patch

        from mcp.shared.memory import create_connected_server_and_client_session

        notifications: list[Any] = []

        async def record(message: Any) -> None:
            notifications.append(message)

        async def fake_run_mcp_tool(
            loop: object, name: str, arguments: dict[str, Any], runtime: object
        ) -> str:
            return f"ran {name}"

        with patch("archex.integrations.mcp._run_mcp_tool", side_effect=fake_run_mcp_tool):
            server = build_server(tool_names=tool_names, disclosure=disclosure)
            async with create_connected_server_and_client_session(
                server, message_handler=record
            ) as client:
                yield client, notifications

    @pytest.mark.asyncio
    async def test_a_fresh_session_is_advertised_only_the_retrieval_core(self) -> None:
        async with self._session(disclosure=True) as (client, _):
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == set(DISCLOSURE_CORE_TOOL_NAMES)

    @pytest.mark.asyncio
    async def test_an_unadvertised_tool_still_dispatches_through_a_closed_gate(
        self,
    ) -> None:
        """The fallback that keeps hardcoded callers working."""
        async with self._session(disclosure=True) as (client, _):
            advertised = {t.name for t in (await client.list_tools()).tools}
            assert "graph_hubs" not in advertised

            result = await client.call_tool("graph_hubs", {"repo_url": "."})
            assert result.isError is False

    @pytest.mark.asyncio
    async def test_retrieving_expands_the_surface_and_tells_the_client(self) -> None:
        from mcp import types as mcp_types

        async with self._session(disclosure=True) as (client, notifications):
            assert len((await client.list_tools()).tools) == len(DISCLOSURE_CORE_TOOL_NAMES)

            await client.call_tool("query_repo", {"repo_url": ".", "question": "q"})

            after = {t.name for t in (await client.list_tools()).tools}
            assert after == set(ALL_TOOL_NAMES)
            assert any(
                isinstance(getattr(n, "root", n), mcp_types.ToolListChangedNotification)
                for n in notifications
            ), "the client was never told its tool list grew"

    @pytest.mark.asyncio
    async def test_an_ungated_session_advertises_everything_immediately(self) -> None:
        async with self._session(disclosure=False) as (client, notifications):
            assert {t.name for t in (await client.list_tools()).tools} == set(ALL_TOOL_NAMES)
            await client.call_tool("query_repo", {"repo_url": ".", "question": "q"})
            assert notifications == [], "nothing changed, so nothing to announce"

    @pytest.mark.asyncio
    async def test_a_disjoint_scope_is_not_announced_because_nothing_changed(
        self,
    ) -> None:
        """`--tools graph` advertises the same five tools before and after."""
        async with self._session(disclosure=True, tool_names=TOOL_SCOPE_PROFILES["graph"]) as (
            client,
            notifications,
        ):
            before = {t.name for t in (await client.list_tools()).tools}
            await client.call_tool("query_repo", {"repo_url": ".", "question": "q"})
            after = {t.name for t in (await client.list_tools()).tools}
            assert before == after == set(TOOL_SCOPE_PROFILES["graph"])
            assert notifications == [], "a pointless tools/list round trip"


class TestClientCompatibilityPath:
    """`install-client --no-disclosure` is the documented path for a client that
    cannot re-fetch its tool list.

    Asserted through the rendered client config rather than the arg builder, so
    these pin what actually lands on disk.
    """

    @staticmethod
    def _args(tool_scope: str | None = None, *, disclosure: bool = True) -> list[str]:
        from archex.client_setup import build_client_install_plan

        plan = build_client_install_plan(
            "claude-code",
            ".",
            scope="project",
            tool_scope=tool_scope,
            disclosure=disclosure,
        )
        content: dict[str, Any] = json.loads(plan.content)
        return content["mcpServers"]["archex"]["args"]

    def test_the_default_config_is_byte_identical_to_the_pre_r5_one(self) -> None:
        """Existing installs must not churn, so the default stays implicit."""
        assert self._args() == ["mcp"]

    def test_the_compatibility_path_writes_the_opt_out(self) -> None:
        assert self._args(disclosure=False) == ["mcp", "--no-disclosure"]

    def test_a_scope_and_the_compatibility_path_compose(self) -> None:
        assert self._args("core", disclosure=False) == [
            "mcp",
            "--tools",
            "core",
            "--no-disclosure",
        ]

    def test_a_scope_alone_is_unchanged_by_r5(self) -> None:
        assert self._args("core") == ["mcp", "--tools", "core"]

    def test_setup_offers_the_same_opt_out_as_install_client(self) -> None:
        """`setup` is the primary onboarding command and the docs promise the flag.

        Without this, `--no-disclosure` existed only on `install-client` while
        `apply_clients_guidance`'s `disclosure` parameter sat unreachable.
        """
        from click.testing import CliRunner

        from archex.cli.main import cli

        result = CliRunner().invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0
        assert "--no-disclosure" in result.output

    def test_setups_opt_out_reaches_the_rendered_config(self) -> None:
        """A flag that parses but never reaches the plan builder is dead code."""
        from pathlib import Path
        from unittest.mock import patch

        from click.testing import CliRunner

        from archex.cli.main import cli
        from archex.cli.setup_cmd import run_preflight

        if not run_preflight(Path(".")).mcp_runtime_available:
            pytest.skip("client planning is skipped when the mcp runtime is unavailable")

        seen: list[bool] = []
        real = None

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(bool(kwargs["disclosure"]))
            assert real is not None
            return real(*args, **kwargs)

        import archex.cli.setup_cmd as setup_mod

        real = setup_mod.build_discovered_install_plans
        with patch.object(setup_mod, "build_discovered_install_plans", side_effect=spy):
            runner = CliRunner()
            runner.invoke(cli, ["setup", ".", "--dry-run", "--clients", "--format", "json"])
            runner.invoke(
                cli,
                ["setup", ".", "--dry-run", "--clients", "--no-disclosure", "--format", "json"],
            )

        assert seen == [True, False], "the flag never reached the plan builder"


class TestTheAcceptanceCommandMeasuresTheShippedDefault:
    """R5's acceptance row names `archex mcp-schema-size --format json` and a
    sub-1000-token result. The command measured scope `all` regardless of the
    gate, so it reported 3859 -- an operator running the documented command saw
    the pre-R5 number and would conclude nothing had improved.
    """

    @staticmethod
    def _json(*args: str) -> dict[str, Any]:
        from click.testing import CliRunner

        from archex.cli.main import cli

        result = CliRunner().invoke(cli, ["mcp-schema-size", *args, "--format", "json"])
        assert result.exit_code == 0, result.output
        parsed: dict[str, Any] = json.loads(result.output)
        return parsed

    def test_the_bare_command_reports_the_gated_cost(self) -> None:
        report = self._json()
        assert report["gated"] is True
        assert report["tool_count"] == len(DISCLOSURE_CORE_TOOL_NAMES)
        assert report["total_tokens"] < DISCLOSURE_TOKEN_BUDGET

    def test_the_gated_report_cannot_hide_the_expanded_cost(self) -> None:
        """765 alone would oversell: a session that retrieves pays the full surface
        afterwards, so the bare command reports both or neither.
        """
        report = self._json()
        expanded = report["after_first_retrieval"]
        assert expanded["tool_count"] == len(ALL_TOOL_NAMES)
        assert expanded["total_tokens"] > report["total_tokens"]

    def test_the_ungated_surface_is_still_reachable(self) -> None:
        report = self._json("--no-disclosure")
        assert report["gated"] is False
        assert report["tool_count"] == len(ALL_TOOL_NAMES)
        assert "after_first_retrieval" not in report

    def test_an_explicit_scope_answers_about_that_scope(self) -> None:
        """`--tools core` asks what `core` costs, not what a gated session sees."""
        report = self._json("--tools", "core")
        assert report["gated"] is False
        assert report["scope"] == "core"
        assert report["tool_count"] == len(TOOL_SCOPE_PROFILES["core"])
