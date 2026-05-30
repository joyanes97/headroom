"""HeadroomEngine — request/response hook facade (Chunk 2 + 4.2a/4.2b).

Composes the existing compression subsystems behind a clean hook interface.
Does NOT reimplement compression; delegates to injected ``CompressionPipeline``
instances via the ``ports.CompressionPipeline`` Protocol.

Design notes
------------
- **Dependency injection**: pipelines, config, usage_reporter are injected;
  no global state is read or written inside this module.
- **No silent fallbacks**: unregistered (provider, flavor) pairs raise loudly.
- **Passthrough fidelity**: when ``CompressionDecision.should_compress`` is
  False, ``on_request`` returns ``ctx.raw_body`` byte-identical (same object,
  no re-serialization).
- **Chunk 4.2a — real Anthropic path**: when ``anthropic_components`` is
  provided the engine orchestrates the full handler compression-core (mode
  branching, frozen-count, tool-sort, prepare_outbound_body_bytes) using the
  SAME callables the handler uses.
- **Chunk 4.2b — CCR request-side**: when ``ccr_components`` is additionally
  provided, the engine runs the full CCR request-side pipeline (workspace
  resolution, marker scan + session-sticky tool injection, compression tracking,
  proactive expansion) AFTER compression-core, using the SAME callables the
  handler uses. Memory injection (4.2c) runs between compression tracking and
  proactive expansion — the ordering seam is clearly marked.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from headroom.engine.contract import (
    Flavor,
    Provider,
    RequestContext,
    RequestDecision,
    ResponseTelemetry,
    StreamContext,
)
from headroom.engine.ports import CompressionPipeline
from headroom.proxy.auth_mode import classify_auth_mode
from headroom.proxy.compression_decision import CompressionDecision
from headroom.transforms.compression_policy import resolve_policy

logger = logging.getLogger("headroom.engine")


class AnthropicComponents:
    """Real Anthropic compression components for the engine.

    Replaces the fake-pipeline-only path when the engine should reproduce
    byte-identical output with the handler's compression-core path.

    Parameters
    ----------
    pipeline:
        The real ``TransformPipeline`` for Anthropic (same object the
        server builds in HeadroomProxy.__init__).
    provider:
        The AnthropicProvider (used for ``get_context_limit``).
    session_tracker_store:
        The ``SessionTrackerStore`` the engine owns (separate from the
        server's store so prefix-tracker state is engine-private).
    get_compression_cache:
        Callable ``(session_id: str) -> CompressionCache`` — same
        semantics as ``HeadroomProxy._get_compression_cache``.
    config:
        The ``ProxyConfig`` (mode, optimize, hooks, …).
    usage_reporter:
        Commercial gate for ``CompressionDecision.decide``.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        provider: Any,
        session_tracker_store: Any,
        get_compression_cache: Callable[[str], Any],
        config: Any,
        usage_reporter: Any | None,
    ) -> None:
        self.pipeline = pipeline
        self.provider = provider
        self.session_tracker_store = session_tracker_store
        self.get_compression_cache = get_compression_cache
        self.config = config
        self.usage_reporter = usage_reporter


class CCRComponents:
    """Injectable CCR request-side components for the engine (Chunk 4.2b).

    When this is provided to ``HeadroomEngine``, the engine runs the full
    CCR request-side pipeline after compression-core. When ``None``, the
    CCR steps are skipped entirely (no-op), which preserves byte-identical
    output for all existing CCR-OFF fixtures.

    Parameters
    ----------
    ccr_context_tracker:
        Live ``CCRContextTracker`` instance, or ``None`` when CCR context
        tracking is disabled. When ``None``, compression-tracking and
        proactive-expansion steps are skipped.
    get_compression_store:
        Callable ``() -> CompressionStore`` — returns the store used for
        ``get_metadata(hash_key)`` lookups during compression tracking.
        Injected so tests can supply a stub store without touching the global.
    turn_counter:
        Mutable single-element list ``[int]`` used to share the turn counter
        across CCR steps. The engine increments ``turn_counter[0]`` when
        compression tracking fires. Callers pass ``[0]`` on first use;
        the engine mutates in place so the counter accumulates across turns
        within the same session (same behaviour as ``self._turn_counter``
        on the proxy handler). Pass a new ``[0]`` per engine instance.
    """

    def __init__(
        self,
        *,
        ccr_context_tracker: Any | None,
        get_compression_store: Callable[[], Any],
        turn_counter: list[int],
    ) -> None:
        self.ccr_context_tracker = ccr_context_tracker
        self.get_compression_store = get_compression_store
        # Mutable single-element list so tests/callers can inspect the count
        # after engine invocations. The engine mutates turn_counter[0] in place.
        self.turn_counter = turn_counter


class HeadroomEngine:
    """Facade that composes Headroom compression behind hook-shaped entry points.

    ``on_request`` is the load-bearing method. Two operating modes:

    **Fake-pipeline mode** (Chunks 1-2 tests, legacy): ``anthropic_components``
    is None; the engine uses ``pipelines`` to dispatch and applies a simplified
    (non-mode-branching) pipeline call. Existing Chunk 2 tests continue to pass
    because this path is unchanged.

    **Real-Anthropic mode** (Chunk 4.2a/4.2b): ``anthropic_components`` is set.
    The engine owns the full compression-core orchestration for Anthropic
    requests: mode-branching (token/non-cache/cache-delta), frozen-count
    derivation, tool-sort, and ``prepare_outbound_body_bytes``. It faithfully
    reproduces what ``AnthropicHandlerMixin.handle_messages`` does for
    compression-core. When ``ccr_components`` is also provided, the CCR
    request-side pipeline (steps 1-4) runs after compression-core.

    Parameters
    ----------
    pipelines:
        Mapping from ``(Provider, Flavor)`` to a ``CompressionPipeline``
        implementor.  Fakes satisfy this in tests; used by the legacy path.
    config:
        Config object forwarded verbatim to ``CompressionDecision.decide``.
        Only ``config.optimize: bool`` is read there.
    usage_reporter:
        Commercial gate forwarded to ``CompressionDecision.decide``.
        ``None`` means no licensing → always allow compression.
    salt:
        Salt bytes for session key derivation (kept for CCR proactive-expansion
        wiring; not consumed in current chunks).
    anthropic_components:
        When set, the engine uses the real Anthropic orchestration path for
        Anthropic/Messages requests (Chunk 4.2a). When None, falls back to
        the fake-pipeline path (Chunks 1-2 behaviour).
    ccr_components:
        When set (and anthropic_components is also set), the engine runs the
        full CCR request-side pipeline after compression-core (Chunk 4.2b).
        When None, CCR steps are a no-op — existing CCR-OFF tests are unchanged.
    """

    def __init__(
        self,
        *,
        pipelines: Mapping[tuple[Provider, Flavor], CompressionPipeline],
        config: Any,
        usage_reporter: Any | None,
        salt: bytes,
        anthropic_components: AnthropicComponents | None = None,
        ccr_components: CCRComponents | None = None,
    ) -> None:
        self._pipelines = dict(pipelines)
        self._config = config
        self._usage_reporter = usage_reporter
        self._salt = salt
        self._anthropic_components = anthropic_components
        self._ccr_components = ccr_components

    # ── Request hook ──────────────────────────────────────────────────────────

    def on_request(self, ctx: RequestContext) -> RequestDecision:
        """Process an inbound request.

        For registered ``(provider, flavor)`` combos: classify auth mode,
        decide whether to compress, and either return the raw body unchanged
        (passthrough) or run the pipeline and return the mutated body.

        Raises
        ------
        KeyError
            If ``(ctx.provider, ctx.flavor)`` has no registered pipeline
            AND no real-component path handles it.
        ValueError
            If the raw body cannot be parsed as JSON (malformed request).
        """
        # Real Anthropic path (Chunk 4.2a + 4.2b)
        if (
            ctx.provider == Provider.ANTHROPIC
            and ctx.flavor == Flavor.MESSAGES
            and self._anthropic_components is not None
        ):
            return self._on_request_anthropic_real(ctx)

        # Legacy fake-pipeline path (Chunks 1-2)
        key = (ctx.provider, ctx.flavor)
        if key not in self._pipelines:
            raise KeyError(
                f"No pipeline registered for provider={ctx.provider!r}, "
                f"flavor={ctx.flavor!r}. Register it in the pipelines mapping."
            )

        return self._on_request_fake_pipeline(ctx, self._pipelines[key])

    # ── Real Anthropic orchestration (Chunk 4.2a + 4.2b) ─────────────────────

    def _on_request_anthropic_real(self, ctx: RequestContext) -> RequestDecision:
        """Reproduce the handler's compression-core + CCR request-side path.

        Mirrors ``AnthropicHandlerMixin.handle_messages`` through:
          image compress → CompressionDecision → mode-branch pipeline.apply →
          tool-sort → [CCR: workspace-resolve, marker-scan, tool-inject,
          system-instruction-inject, compression-tracking, proactive-expansion]
          → prepare_outbound_body_bytes.

        CCR steps are a no-op when ``self._ccr_components`` is None (all
        existing CCR-OFF fixtures remain byte-identical).

        Ordering seam for 4.2c (memory injection):
            Memory injection runs AFTER compression tracking (step 3) and
            BEFORE proactive expansion (step 4). The comment marked
            ``# ── 4.2c SEAM: memory injection goes here ──`` is the exact
            insertion point. Do NOT move step 4 above that comment.

        Excluded from this chunk: hooks, pipeline_extension events, security
        scan, traffic_learner, memory injection (4.2c).
        """
        from headroom.cache.compression_cache import CompressionCache  # noqa: F401
        from headroom.proxy.helpers import (
            BodyMutationTracker,
            prepare_outbound_body_bytes,
        )
        from headroom.proxy.image_compression_decision import ImageCompressionDecision
        from headroom.proxy.modes import is_cache_mode, is_token_mode
        from headroom.utils import extract_user_query

        ac = self._anthropic_components
        assert ac is not None

        original_body_bytes = ctx.raw_body

        # Parse body — raises loudly on malformed JSON
        try:
            body: dict[str, Any] = json.loads(original_body_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"on_request(anthropic): unparseable JSON body: {exc}") from exc

        messages: list[dict[str, Any]] = list(body.get("messages") or [])
        model: str = body.get("model", "unknown")
        # Preserve a deep copy of the original client messages (mirrors deep_copy
        # at handler line ~595) for use in the cache-delta path.
        original_client_messages: list[dict[str, Any]] = copy.deepcopy(messages)

        # Bypass: skip ALL compression when the caller explicitly opts out.
        headers = dict(ctx.headers_view)
        _bypass = (
            headers.get("x-headroom-bypass", "").lower() == "true"
            or headers.get("x-headroom-mode", "").lower() == "passthrough"
        )

        body_mutation_tracker = BodyMutationTracker()

        # Auth mode + policy (computed once; used by all three pipeline sites)
        auth_mode = classify_auth_mode(ctx.headers_view)
        compression_policy = resolve_policy(auth_mode)

        # Compression decision
        _decision = CompressionDecision.decide(
            headers=ctx.headers_view,
            config=ac.config,
            usage_reporter=ac.usage_reporter,
            messages=messages,
        )

        if not _decision.should_compress or _bypass:
            # Passthrough — return original bytes byte-identical.
            return RequestDecision(
                body=original_body_bytes,
                telemetry=ResponseTelemetry(compressed=False),
            )

        # --- Image compression (before text compression, same order as handler) ---
        _image_decision = ImageCompressionDecision.decide(
            headers=ctx.headers_view, config=ac.config, messages=messages
        )
        if _image_decision.should_compress and not is_cache_mode(ac.config.mode):
            from headroom.proxy.helpers import _get_image_compressor

            compressor = None
            try:
                compressor = _get_image_compressor()
                if compressor and compressor.has_images(messages):
                    messages = compressor.compress(messages, provider="anthropic")
                    body_mutation_tracker.mark_mutated("image_compression")
            finally:
                if compressor and hasattr(compressor, "close"):
                    compressor.close()

        # --- Session / frozen-count derivation ---
        # The engine owns its own session store (injected via AnthropicComponents);
        # the parity test seeds it with a controlled _FixedTracker just as the
        # golden recorder does.
        session_id = ac.session_tracker_store.compute_session_id(ctx, model, messages)
        prefix_tracker = ac.session_tracker_store.get_or_create(session_id, "anthropic")
        frozen_message_count = prefix_tracker.get_frozen_message_count()
        if is_cache_mode(ac.config.mode):
            # Mirrors _strict_previous_turn_frozen_count at handler line ~890.
            frozen_message_count = _strict_previous_turn_frozen_count(
                original_client_messages, frozen_message_count
            )

        # --- Context limit ---
        context_limit = ac.provider.get_context_limit(model)

        # --- hooks/biases (skipped in 4.2a — not present in golden corpus) ---
        biases = None
        request_id = ctx.request_id

        optimized_messages = messages

        # --- Mode branch: token / non-cache / cache-delta ---
        if is_token_mode(ac.config.mode):
            comp_cache = ac.get_compression_cache(session_id)

            # Zone 1: swap cached compressed versions into working copy
            working_messages = comp_cache.apply_cached(messages)

            # Clamp frozen_message_count (mirrors handler lines ~1039-1042)
            cache_frozen_count = comp_cache.compute_frozen_count(messages)
            frozen_message_count = min(frozen_message_count, cache_frozen_count)
            comp_cache.mark_stable_from_messages(messages, frozen_message_count)

            result = ac.pipeline.apply(
                messages=working_messages,
                model=model,
                model_limit=context_limit,
                context=extract_user_query(working_messages),
                frozen_message_count=frozen_message_count,
                biases=biases,
                request_id=request_id,
                compression_policy=compression_policy,
            )

            if result.messages != working_messages:
                comp_cache.update_from_result(messages, result.messages)

            optimized_messages = result.messages
            # Mirror handler line ~1064: always use pipeline result.
            # Structural diff check below detects any real mutation.

        elif not is_cache_mode(ac.config.mode):
            result = ac.pipeline.apply(
                messages=messages,
                model=model,
                model_limit=context_limit,
                context=extract_user_query(messages),
                frozen_message_count=frozen_message_count,
                biases=biases,
                request_id=request_id,
                compression_policy=compression_policy,
            )

            if result.messages != messages:
                optimized_messages = result.messages
                # Do NOT mark mutation explicitly here; structural diff below
                # detects the actual byte change. Handler mirrors this: no
                # explicit mark at lines ~1099-1104 for the non-cache path.

        else:
            # Cache-delta path
            previous_original_messages = prefix_tracker.get_last_original_messages()
            previous_forwarded_messages = prefix_tracker.get_last_forwarded_messages()
            delta = _extract_cache_stable_delta(
                original_client_messages,
                previous_original_messages,
                previous_forwarded_messages,
            )
            if delta is not None:
                stable_forwarded_prefix, delta_messages = delta
                if delta_messages:
                    result = ac.pipeline.apply(
                        messages=delta_messages,
                        model=model,
                        model_limit=context_limit,
                        context=extract_user_query(delta_messages),
                        frozen_message_count=0,
                        biases=biases,
                        request_id=request_id,
                        compression_policy=compression_policy,
                    )
                    optimized_messages = stable_forwarded_prefix + result.messages
                    # Mirror the handler: no explicit mark_mutated here.
                    # The structural diff check below will detect any real change.
                else:
                    optimized_messages = stable_forwarded_prefix
                    # No explicit mutation mark — structural diff detects if needed.
            else:
                # Conservative fallback for cache mode
                optimized_messages = messages

        # --- Tool sort (ALWAYS when tools present) ---
        tools = body.get("tools")
        if tools is not None:
            from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

            sorted_tools = AnthropicHandlerMixin._sort_tools_deterministically(tools)
            if sorted_tools != tools:
                body_mutation_tracker.mark_mutated("tool_sort")
            body["tools"] = sorted_tools
            tools = body["tools"]  # keep local alias in sync

        # ── CCR request-side (Chunk 4.2b) ────────────────────────────────────
        # Steps 1-4 are a no-op when ccr_components is None or CCR config flags
        # are off, preserving byte-identical output for all existing CCR-OFF
        # fixtures.
        ccr_tool_injected = False
        ccr_workspace_key = ""
        ccr_workspace_label: str | None = None

        ccr = self._ccr_components
        if ccr is not None and not _bypass:
            # Step 1: workspace resolution ────────────────────────────────────
            # Adapted from AnthropicHandlerMixin._resolve_ccr_workspace to work
            # from (headers_view, body) instead of a FastAPI Request object.
            # Fail-closed: ("", None) → skip CCR tracking + expansion.
            ccr_workspace_key, ccr_workspace_label = _resolve_ccr_workspace(ctx.headers_view, body)

            # Step 2: marker scan + session-sticky tool injection ─────────────
            # Gated on the same config flags as the handler.
            if ac.config.ccr_inject_tool or ac.config.ccr_inject_system_instructions:
                from headroom.ccr import CCRToolInjector
                from headroom.proxy.helpers import apply_session_sticky_ccr_tool

                inject_system_instructions = ac.config.ccr_inject_system_instructions
                if inject_system_instructions and frozen_message_count > 0:
                    # Cache hot zone — skip to preserve prefix cache bytes.
                    logger.info(
                        "[%s] CCR(engine): skipping system instruction injection "
                        "(frozen prefix=%d) to preserve cache",
                        request_id,
                        frozen_message_count,
                    )
                    inject_system_instructions = False

                inject_tool = ac.config.ccr_inject_tool
                if inject_tool and frozen_message_count > 0:
                    logger.info(
                        "[%s] CCR(engine): deferring tool injection "
                        "(frozen prefix=%d) to preserve cache",
                        request_id,
                        frozen_message_count,
                    )
                    inject_tool = False

                # Scan for compression markers; always with inject_tool=False
                # because tool-list injection goes through the sticky helper.
                injector = CCRToolInjector(
                    provider="anthropic",
                    inject_tool=False,
                    inject_system_instructions=inject_system_instructions,
                )
                injector.scan_for_markers(optimized_messages)

                # System-instruction injection: only when frozen==0 and compressed.
                if inject_system_instructions and injector.has_compressed_content:
                    optimized_messages = injector.inject_into_system_message(optimized_messages)
                    body_mutation_tracker.mark_mutated("ccr_system_instruction_inject")

                # Sticky-on tool registration (PR-B7): once a session has done CCR
                # the retrieve tool stays in body["tools"] every turn.
                if inject_tool:
                    tools, ccr_tool_injected = apply_session_sticky_ccr_tool(
                        provider="anthropic",
                        session_id=session_id,
                        request_id=request_id,
                        existing_tools=tools,
                        has_compressed_content_this_turn=injector.has_compressed_content,
                    )
                    if ccr_tool_injected:
                        body["tools"] = tools
                        body_mutation_tracker.mark_mutated("ccr_tool_inject")
                        logger.debug(
                            "[%s] CCR(engine): tool registered (session=%s, "
                            "compressed_this_turn=%s, hashes_seen=%d)",
                            request_id,
                            session_id,
                            injector.has_compressed_content,
                            len(injector.detected_hashes),
                        )

                # Step 3: compression tracking ────────────────────────────────
                # Gated on: has_compressed_content AND ccr_context_tracker AND
                # workspace_key resolved. Fail-closed when workspace is empty —
                # tracked under empty key would be un-matchable in analyze_query.
                if injector.has_compressed_content:
                    if ccr.ccr_context_tracker and ccr_workspace_key:
                        ccr.turn_counter[0] += 1
                        for hash_key in injector.detected_hashes:
                            store = ccr.get_compression_store()
                            entry = store.get_metadata(hash_key)
                            if entry:
                                ccr.ccr_context_tracker.track_compression(
                                    hash_key=hash_key,
                                    turn_number=ccr.turn_counter[0],
                                    tool_name=entry.get("tool_name"),
                                    original_count=entry.get("original_item_count", 0),
                                    compressed_count=entry.get("compressed_item_count", 0),
                                    workspace_key=ccr_workspace_key,
                                    query_context=entry.get("query_context", ""),
                                    sample_content=entry.get("compressed_content", "")[:500],
                                )
                    elif ccr.ccr_context_tracker and not ccr_workspace_key:
                        # Explicit fail-closed log — not a silent skip.
                        logger.info(
                            "[%s] CCR(engine): workspace unresolved; skipping "
                            "track_compression (fail-closed — no x-headroom-cwd / "
                            "x-headroom-project-id header and no cwd: in system prompt)",
                            request_id,
                        )

        # ── 4.2c SEAM: memory injection goes here ────────────────────────────
        # Memory injection (Chunk 4.2c) must run AFTER compression tracking
        # (step 3 above) and BEFORE proactive expansion (step 4 below).
        # Insert the memory injection call at this exact location. The
        # injected context should be appended to the latest non-frozen user
        # turn via AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn.
        # ─────────────────────────────────────────────────────────────────────

        # Step 4: proactive expansion ─────────────────────────────────────────
        # ORDERING NOTE: In the full handler this runs AFTER memory injection.
        # The 4.2c memory seam above is the canonical insertion point.
        # Gated on the same workspace and config flags as the handler.
        if (
            ccr is not None
            and not _bypass
            and ccr.ccr_context_tracker is not None
            and ac.config.ccr_proactive_expansion
            and ccr_workspace_key
        ):
            from headroom.proxy.modes import is_cache_mode as _is_cache_mode

            # Extract user query from messages (same loop as handler lines ~1340-1351).
            user_query = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_query = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                user_query = block.get("text", "")
                                break
                    break

            if user_query:
                recommendations = ccr.ccr_context_tracker.analyze_query(
                    user_query,
                    ccr.turn_counter[0],
                    workspace_key=ccr_workspace_key,
                )
                if recommendations:
                    expansions = ccr.ccr_context_tracker.execute_expansions(recommendations)
                    if expansions:
                        expansion_text = ccr.ccr_context_tracker.format_expansions_for_context(
                            expansions,
                            workspace_label=ccr_workspace_label,
                        )
                        logger.info(
                            "[%s] CCR(engine): proactively expanded %d context(s) "
                            "based on query relevance",
                            request_id,
                            len(expansions),
                        )
                        if _is_cache_mode(ac.config.mode):
                            logger.info(
                                "[%s] CCR(engine): skipping proactive expansion append "
                                "in cache mode to preserve next-turn prefix stability",
                                request_id,
                            )
                        else:
                            from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

                            optimized_messages = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
                                optimized_messages,
                                expansion_text,
                                frozen_message_count=frozen_message_count,
                            )
                            body_mutation_tracker.mark_mutated("ccr_proactive_expansion")

        # --- Reassemble body ---
        body["messages"] = optimized_messages

        # --- Structural mutation safety-net (mirrors handler lines ~1654-1660) ---
        if not body_mutation_tracker.mutated:
            try:
                parsed_original = json.loads(original_body_bytes)
                if parsed_original != body:
                    body_mutation_tracker.mark_mutated("structural_diff_vs_original")
            except (json.JSONDecodeError, ValueError):
                body_mutation_tracker.mark_mutated("original_unparseable")

        # --- Byte-faithful forward (mirrors prepare_outbound_body_bytes) ---
        outbound_bytes, _source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=original_body_bytes,
            body_mutated=body_mutation_tracker.mutated,
        )

        compressed = body_mutation_tracker.mutated
        bytes_saved = max(0, len(original_body_bytes) - len(outbound_bytes))

        return RequestDecision(
            body=outbound_bytes,
            telemetry=ResponseTelemetry(
                bytes_saved=bytes_saved,
                compressed=compressed,
                ccr_fired=ccr_tool_injected,
            ),
        )

    # ── Legacy fake-pipeline path (Chunks 1-2) ────────────────────────────────

    def _on_request_fake_pipeline(
        self, ctx: RequestContext, pipeline: CompressionPipeline
    ) -> RequestDecision:
        """Simplified path used by Chunk 2 tests with fake pipelines.

        Preserves the original Chunk 2 semantics exactly so those tests
        continue passing.
        """
        # Parse body — raises loudly on malformed JSON
        try:
            body: dict[str, Any] = json.loads(ctx.raw_body)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"on_request: unparseable JSON body for "
                f"provider={ctx.provider!r}, flavor={ctx.flavor!r}: {exc}"
            ) from exc

        messages: list[dict[str, Any]] = body.get("messages") or []
        model: str = body.get("model", "")

        # Classify auth mode (pure, <10us, never raises)
        auth_mode = classify_auth_mode(ctx.headers_view)

        # Decision: should we compress?
        decision = CompressionDecision.decide(
            headers=ctx.headers_view,
            config=self._config,
            usage_reporter=self._usage_reporter,
            messages=messages,
        )

        if not decision.should_compress:
            # Return raw body BYTE-IDENTICAL — same object, no re-serialization.
            # This is load-bearing for prefix-cache safety.
            return RequestDecision(
                body=ctx.raw_body,
                telemetry=ResponseTelemetry(compressed=False),
            )

        # Resolve per-auth-mode compression policy
        policy = resolve_policy(auth_mode)

        # Delegate to the injected pipeline
        result = pipeline.apply(
            messages,
            model,
            compression_policy=policy,
        )

        # Reconstruct body with compressed messages
        body["messages"] = result.messages
        compressed_bytes = json.dumps(body).encode()

        bytes_saved = max(0, len(ctx.raw_body) - len(compressed_bytes))
        tokens_in = getattr(result, "tokens_before", 0)
        tokens_out = getattr(result, "tokens_after", 0)

        return RequestDecision(
            body=compressed_bytes,
            telemetry=ResponseTelemetry(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                bytes_saved=bytes_saved,
                compressed=True,
                ccr_fired=False,
            ),
        )

    # ── Response hooks (Chunk 2 stubs — Chunk 3+ extends these) ─────────────

    def on_response(self, ctx: RequestContext, raw_response: bytes) -> bytes:
        """Forward the upstream response unchanged.

        Chunk 3 will extend this with CCR proactive-expansion injection and
        token telemetry parsing.
        """
        return raw_response

    def on_response_chunk(self, sc: StreamContext, chunk: bytes) -> bytes:
        """Forward a streaming chunk unchanged.

        Chunk 3 will add SSE parsing for streaming token telemetry.
        """
        return chunk

    def on_response_end(self, sc: StreamContext, outcome: Any) -> ResponseTelemetry:
        """Finalize a streaming session and return its telemetry.

        Safe to call on normal completion OR abort (``outcome`` may be an
        Exception or ``None``).  Chunk 3 will accumulate streaming token
        counts here.
        """
        return ResponseTelemetry()


# ── Private helpers (mirrors static methods on AnthropicHandlerMixin) ─────────


def _resolve_ccr_workspace(
    headers_view: Mapping[str, str],
    body: dict[str, Any],
) -> tuple[str, str | None]:
    """Resolve (workspace_key, workspace_label) for CCR scoping.

    Adapted from ``AnthropicHandlerMixin._resolve_ccr_workspace`` to work
    from ``(headers_view, body)`` instead of a FastAPI ``Request`` object.
    The engine has no FastAPI request; all header/body signals are available
    through ``ctx.headers_view`` and the parsed ``body`` dict respectively.

    Tier order is identical to the handler:
      x-headroom-project-id → x-headroom-cwd → CLI override (N/A here,
      project_root_override=None) → cwd: line in system prompt.

    Returns ``("", None)`` on any failure — fail-closed, not silent.
    A warning is logged so the absence is observable.
    """
    from headroom.memory.storage_router import (
        ProjectResolver,
    )
    from headroom.memory.storage_router import (
        RequestContext as _StorageCtx,
    )
    from headroom.memory.storage_router import (
        extract_system_prompt as _extract_sys_prompt,
    )

    try:
        storage_ctx = _StorageCtx(
            headers=dict(headers_view),
            system_prompt=_extract_sys_prompt(body),
            base_user_id=dict(headers_view).get("x-headroom-user-id", ""),
            project_root_override=None,
        )
        ident = ProjectResolver().resolve(storage_ctx)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "event=ccr_workspace_resolve_failed error=%s; "
            "CCR proactive expansion disabled for this request",
            exc,
        )
        return "", None

    if ident is None:
        return "", None
    return ident[0], ident[1]


def _strict_previous_turn_frozen_count(
    messages: list[dict[str, Any]],
    base_frozen_count: int,
) -> int:
    """Freeze all prior turns; only the final turn is mutable.

    Direct port of ``AnthropicHandlerMixin._strict_previous_turn_frozen_count``.
    """
    if not messages:
        return base_frozen_count
    final_idx = len(messages) - 1
    if messages[final_idx].get("role") == "user":
        return max(base_frozen_count, final_idx)
    return len(messages)


def _extract_cache_stable_delta(
    current_messages: list[dict[str, Any]],
    previous_original_messages: list[dict[str, Any]] | None,
    previous_forwarded_messages: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return (stable_forwarded_prefix, appended_delta_messages) when safe.

    Direct port of ``AnthropicHandlerMixin._extract_cache_stable_delta``.
    """
    if not previous_original_messages or previous_forwarded_messages is None:
        return None
    prefix_len = len(previous_original_messages)
    if len(current_messages) < prefix_len:
        return None
    if current_messages[:prefix_len] != previous_original_messages:
        return None
    return (
        copy.deepcopy(previous_forwarded_messages),
        copy.deepcopy(current_messages[prefix_len:]),
    )
