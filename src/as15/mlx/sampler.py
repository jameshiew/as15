# MLX diffusion generation loop for AceStep DiT decoder.
#
# Replicates the timestep scheduling and ODE/SDE stepping from
# ``AceStepConditionGenerationModel.generate_audio`` using pure MLX arrays.
#
# Enhanced sampling modes (see issue #957):
# - ``euler``: First-order Euler ODE/SDE step (default, original behaviour).
# - ``heun``: Second-order Heun predictor-corrector -- evaluates the model
#   twice per step and averages the predictions for higher accuracy, which
#   matters especially with 8-step turbo inference.
#
# Optional stabilisation techniques (work with *any* sampler mode):
# - ``velocity_norm_threshold``: Clamp the L2 norm of velocity predictions
#   relative to the input norm.  Prevents outlier predictions that cause
#   audio artefacts.
# - ``velocity_ema_factor``: Exponential moving average blending between
#   the current and previous velocity prediction, smoothing the denoising
#   trajectory.

import logging
import math
import time
from typing import TypedDict

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DiffusionResult(TypedDict):
    """Return shape of :func:`mlx_generate_diffusion`.

    ``time_costs`` mixes float durations with the ``sampler_mode`` label, hence
    the widened value type.
    """

    target_latents: np.ndarray
    time_costs: dict[str, float | str]


VALID_SAMPLER_MODES = {"euler", "heun"}
VALID_INFER_METHODS = {"ode", "sde"}

# Momentum coefficient of the APG running average, from the default of
# upstream's ``MomentumBuffer`` (apg_guidance.py). Negative: each step
# subtracts three quarters of the previous guidance difference rather than
# accumulating it.
APG_MOMENTUM = -0.75

# Pre-defined timestep schedules (from modeling_acestep_v15_turbo.py)
VALID_SHIFTS = [1.0, 2.0, 3.0]

VALID_TIMESTEPS = [
    1.0,
    0.9545454545454546,
    0.9333333333333333,
    0.9,
    0.875,
    0.8571428571428571,
    0.8333333333333334,
    0.7692307692307693,
    0.75,
    0.6666666666666666,
    0.6428571428571429,
    0.625,
    0.5454545454545454,
    0.5,
    0.4,
    0.375,
    0.3,
    0.25,
    0.2222222222222222,
    0.125,
]

SHIFT_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [
        1.0,
        0.9333333333333333,
        0.8571428571428571,
        0.7692307692307693,
        0.6666666666666666,
        0.5454545454545454,
        0.4,
        0.2222222222222222,
    ],
    3.0: [
        1.0,
        0.9545454545454546,
        0.9,
        0.8333333333333334,
        0.75,
        0.6428571428571429,
        0.5,
        0.3,
    ],
}


def check_schedule_options(shift: float, infer_steps: int | None = None) -> None:
    """Reject step counts and shifts the schedule cannot express.

    Both used to be accepted and then quietly mean something else. A step count
    below 1 fell through to the fixed 8-step table, so the CLI reported the
    number it was asked for while the loop ran eight steps; and shift=0 divided
    0 by 0 at t=1 (the map's denominator is ``1+(shift-1)*t``), while a negative
    shift either hit that same zero denominator part-way down the schedule or
    produced a non-monotonic one. Nothing downstream can tell those apart from a
    deliberate setting.
    """
    if infer_steps is not None and infer_steps < 1:
        raise ValueError(f"infer_steps must be at least 1, got {infer_steps}.")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError(f"shift must be finite and greater than zero, got {shift}.")


def check_sampling_options(
    sampler_mode: str,
    infer_method: str,
    shift: float,
    infer_steps: int | None = None,
) -> None:
    """Reject an unusable sampler configuration.

    Called by :func:`mlx_generate_diffusion`, and separately by the pipeline
    before it fetches snapshots and runs conditioning -- otherwise a typo in a
    request surfaces minutes later, on entry to the diffusion loop.
    """
    if sampler_mode not in VALID_SAMPLER_MODES:
        raise ValueError(
            f"Unsupported sampler_mode '{sampler_mode}'. Expected one of {sorted(VALID_SAMPLER_MODES)}."
        )
    # An unrecognised method used to fall through to the ODE branch, so a typo
    # ran a different sampler than the caller asked for and nothing said so.
    if infer_method not in VALID_INFER_METHODS:
        raise ValueError(
            f"Unsupported infer_method '{infer_method}'. Expected one of {sorted(VALID_INFER_METHODS)}."
        )
    check_schedule_options(shift, infer_steps)


def get_timestep_schedule(
    shift: float = 3.0,
    timesteps: list | None = None,
    infer_steps: int | None = None,
) -> list[float]:
    """Compute the timestep schedule for diffusion sampling.

    When ``infer_steps`` is provided and ``timesteps`` is None, a continuous
    linspace schedule is generated (matching the PyTorch base-model behaviour).
    The legacy lookup-table path (8-step ``SHIFT_TIMESTEPS``) is used only when
    neither ``timesteps`` nor ``infer_steps`` is supplied.

    Args:
        shift: Diffusion timestep shift (applied via ``shift*t / (1+(shift-1)*t)``).
        timesteps: Optional custom list of timesteps.
        infer_steps: Number of diffusion steps.  When given, overrides the
            fixed 8-step lookup table.

    Returns:
        List of timestep values (descending, without trailing 0).

    Raises:
        ValueError: If ``infer_steps`` is below 1, or ``shift`` is not a finite
            positive number.
    """
    check_schedule_options(shift, infer_steps)

    t_schedule_list = None

    if timesteps is not None:
        ts_list = list(timesteps)
        while ts_list and ts_list[-1] == 0:
            ts_list.pop()
        if len(ts_list) < 1:
            logger.warning(
                "timesteps empty after removing zeros; using default shift=%s", shift
            )
        else:
            if len(ts_list) > 20:
                logger.warning("timesteps length=%d > 20; truncating", len(ts_list))
                ts_list = ts_list[:20]
            mapped = [
                min(VALID_TIMESTEPS, key=lambda x, t=t: abs(x - t)) for t in ts_list
            ]
            t_schedule_list = mapped

    if t_schedule_list is None and infer_steps is not None:
        raw = [1.0 - i / infer_steps for i in range(infer_steps)]
        if shift != 1.0:
            raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
        t_schedule_list = raw

    if t_schedule_list is None:
        original_shift = shift
        shift = min(VALID_SHIFTS, key=lambda x: abs(x - shift))
        if original_shift != shift:
            logger.warning(
                "shift=%.2f rounded to nearest valid shift=%.1f", original_shift, shift
            )
        t_schedule_list = SHIFT_TIMESTEPS[shift]

    return t_schedule_list


def _mlx_apg_forward(
    pred_cond,
    pred_uncond,
    guidance_scale: float,
    momentum_state: dict | None = None,
    norm_threshold: float = 2.5,
):
    """APG (Adaptive Projected Guidance) in pure MLX -- mirrors the PyTorch ``apg_forward``.

    Projection is performed along axis 1 (the time/sequence dimension) to match
    the PyTorch implementation which calls ``apg_forward(..., dims=[1])``.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)

    proj_axis = 1

    diff = pred_cond - pred_uncond
    if momentum_state is not None:
        diff = diff + APG_MOMENTUM * momentum_state.get("running", 0)
        momentum_state["running"] = diff

    if norm_threshold > 0:
        diff_norm = mx.sqrt((diff * diff).sum(axis=proj_axis, keepdims=True))
        scale_factor = mx.minimum(
            mx.ones_like(diff_norm), norm_threshold / (diff_norm + 1e-8)
        )
        diff = diff * scale_factor

    v1 = pred_cond / (
        mx.sqrt((pred_cond * pred_cond).sum(axis=proj_axis, keepdims=True)) + 1e-8
    )
    parallel = (diff * v1).sum(axis=proj_axis, keepdims=True) * v1
    orthogonal = diff - parallel

    return pred_cond + (guidance_scale - 1) * orthogonal


def _mlx_cfg_forward(pred_cond, pred_uncond, guidance_scale: float):
    """Plain classifier-free guidance -- mirrors upstream's ``cfg_forward``.

    The Heun corrector guides through this rather than through APG: it is a
    second evaluation *within* one step, and APG's momentum recurrence has to
    advance once per step.
    """
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


def _mlx_repaint_step_injection(xt, clean_src, mask, t_next, noise):
    """Replace non-repaint regions of *xt* with noised source latents (MLX)."""
    import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)

    zt = t_next * noise + (1.0 - t_next) * clean_src
    m = mx.expand_dims(mask, axis=-1)
    return mx.where(m, xt, zt)


def _mlx_repaint_boundary_blend(x_gen, clean_src, mask_np, cf_frames):
    """Blend generated latents with source at repaint boundaries (MLX)."""
    import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)

    soft = mask_np.astype(np.float32).copy()
    if cf_frames <= 0:
        m = mx.expand_dims(mx.array(soft), axis=-1)
        return m * x_gen + (1.0 - m) * clean_src
    B, T = mask_np.shape
    for b in range(B):
        row = mask_np[b]
        if row.all() or not row.any():
            continue
        idx = np.nonzero(row)[0]
        if len(idx) == 0:
            continue
        left, right = int(idx[0]), int(idx[-1]) + 1
        fs = max(left - cf_frames, 0)
        if left - fs > 0:
            soft[b, fs:left] = np.linspace(0, 1, left - fs + 2)[1:-1]
        fe = min(right + cf_frames, T)
        if fe - right > 0:
            soft[b, right:fe] = np.linspace(1, 0, fe - right + 2)[1:-1]
    m = mx.expand_dims(mx.array(soft), axis=-1)
    return m * x_gen + (1.0 - m) * clean_src


def mlx_generate_diffusion(
    mlx_decoder,
    encoder_hidden_states_np: np.ndarray,
    context_latents_np: np.ndarray,
    src_latents_shape: tuple[int, ...],
    seed: int | list[int] | None = None,
    infer_method: str = "ode",
    shift: float = 3.0,
    timesteps: list | None = None,
    infer_steps: int | None = None,
    guidance_scale: float = 1.0,
    null_condition_emb_np: np.ndarray | None = None,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    audio_cover_strength: float = 1.0,
    encoder_hidden_states_non_cover_np: np.ndarray | None = None,
    context_latents_non_cover_np: np.ndarray | None = None,
    retake_seed: int | list[int] | None = None,
    retake_variance: float = 0.0,
    compile_model: bool = False,
    disable_tqdm: bool = False,
    sampler_mode: str = "euler",
    velocity_norm_threshold: float = 0.0,
    velocity_ema_factor: float = 0.0,
    dcw_enabled: bool = True,
    dcw_mode: str = "double",
    dcw_scaler: float = 0.05,
    dcw_high_scaler: float = 0.02,
    compute_dtype: str = "bfloat16",
    repaint_mask_np: np.ndarray | None = None,
    clean_src_latents_np: np.ndarray | None = None,
    repaint_crossfade_frames: int = 10,
    repaint_injection_ratio: float = 0.5,
) -> DiffusionResult:
    """Run the complete MLX diffusion loop with optional CFG guidance.

    This is the core generation function.  It accepts numpy arrays (converted
    from PyTorch tensors by the handler) and returns numpy arrays that the
    handler converts back to PyTorch.

    Args:
        mlx_decoder: ``MLXDiTDecoder`` instance with loaded weights.
        encoder_hidden_states_np: [B, enc_L, D] from prepare_condition (numpy).
        context_latents_np: [B, T, C] from prepare_condition (numpy).
        src_latents_shape: shape tuple [B, T, 64] for noise generation.
        seed: random seed (int, list[int], or None).
        infer_method: "ode" or "sde".
        shift: timestep shift factor.
        timesteps: optional custom timestep list.
        infer_steps: number of diffusion steps.
        guidance_scale: CFG guidance strength (>1.0 enables CFG).
        null_condition_emb_np: [1, 1, D] null condition embedding for CFG.
        cfg_interval_start: timestep ratio below which CFG is disabled.
        cfg_interval_end: timestep ratio above which CFG is disabled.
        audio_cover_strength: cover strength (0-1).
        encoder_hidden_states_non_cover_np: optional [B, enc_L, D] for non-cover.
        context_latents_non_cover_np: optional [B, T, C] for non-cover.
        compile_model: If True, compile the decoder step with ``mx.compile``.
        disable_tqdm: If True, suppress the diffusion progress bar.
        sampler_mode: Sampler algorithm -- ``"euler"`` (first-order, default) or
            ``"heun"`` (second-order predictor-corrector for cleaner output).
        velocity_norm_threshold: Clamp velocity prediction L2 norm relative to
            input norm at each step.  0 disables (default).  Values around
            1.5-3.0 reduce outlier artefacts.
        velocity_ema_factor: Blend current velocity prediction with the previous
            step's prediction via EMA (``vt = (1-f)*vt + f*prev``).
            0 disables (default).  Values around 0.05-0.2 smooth the trajectory.

    Returns:
        Dict with ``"target_latents"`` (numpy) and ``"time_costs"`` dict.
        ``time_costs["sampler_mode"]`` names the sampler that ran, which is
        ``"euler"`` when a Heun request degraded under SDE.

    Raises:
        ValueError: If ``sampler_mode``, ``infer_method``, ``infer_steps`` or
            ``shift`` is outside what this loop supports.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)

    from .dit import MLXCrossAttentionCache

    check_sampling_options(sampler_mode, infer_method, shift, infer_steps)

    # Heun's corrector is an ODE construction; under SDE every step is Euler.
    # Track the sampler that will actually run, so the warning below and the
    # ``sampler_mode`` reported in ``time_costs`` agree with each other.
    effective_sampler_mode = sampler_mode
    if sampler_mode == "heun" and infer_method == "sde":
        logger.warning(
            "[MLX-DiT] Heun sampler is not supported with SDE inference method. "
            "Falling back to Euler. Use infer_method='ode' for Heun."
        )
        effective_sampler_mode = "euler"

    use_heun = effective_sampler_mode == "heun"
    use_norm_clamp = velocity_norm_threshold > 0
    use_ema = velocity_ema_factor > 0

    if use_heun:
        logger.info(
            "[MLX-DiT] Using Heun (second-order) sampler for higher-quality output."
        )
    if use_norm_clamp:
        logger.info(
            "[MLX-DiT] Velocity norm clamping enabled (threshold=%.2f).",
            velocity_norm_threshold,
        )
    if use_ema:
        logger.info(
            "[MLX-DiT] Velocity EMA smoothing enabled (factor=%.3f).",
            velocity_ema_factor,
        )

    time_costs: dict[str, float | str] = {}
    total_start = time.time()

    # Run the loop in the same dtype as the decoder weights. Feeding fp32
    # activations to bf16 weights makes MLX promote the weights per-op, which
    # costs both bandwidth and peak memory on a 4B model.
    dt = getattr(mx, compute_dtype)

    enc_hs = mx.array(encoder_hidden_states_np).astype(dt)
    ctx = mx.array(context_latents_np).astype(dt)

    enc_hs_nc = (
        mx.array(encoder_hidden_states_non_cover_np).astype(dt)
        if encoder_hidden_states_non_cover_np is not None
        else None
    )
    ctx_nc = (
        mx.array(context_latents_non_cover_np).astype(dt)
        if context_latents_non_cover_np is not None
        else None
    )

    # ---- Repaint setup ----
    do_repaint = repaint_mask_np is not None and clean_src_latents_np is not None
    repaint_mask_mx = mx.array(repaint_mask_np) if do_repaint else None
    clean_src_mx = mx.array(clean_src_latents_np).astype(dt) if do_repaint else None

    bsz = src_latents_shape[0]
    T = src_latents_shape[1]
    C = src_latents_shape[2]

    # ---- CFG setup ----
    do_cfg = guidance_scale > 1.0 and null_condition_emb_np is not None
    null_cond = mx.array(null_condition_emb_np).astype(dt) if do_cfg else None
    if do_cfg:
        null_expanded = mx.broadcast_to(null_cond, enc_hs.shape)
        enc_hs = mx.concatenate([enc_hs, null_expanded], axis=0)
        ctx = mx.concatenate([ctx, ctx], axis=0)
        if enc_hs_nc is not None:
            null_expanded_nc = mx.broadcast_to(null_cond, enc_hs_nc.shape)
            enc_hs_nc = mx.concatenate([enc_hs_nc, null_expanded_nc], axis=0)
        if ctx_nc is not None:
            ctx_nc = mx.concatenate([ctx_nc, ctx_nc], axis=0)
    momentum_state: dict | None = {} if do_cfg else None

    # ---- Noise preparation ----
    def _draw_noise(_seed):
        if _seed is None:
            return mx.random.normal((bsz, T, C)).astype(dt)
        if isinstance(_seed, list):
            parts = []
            for s in _seed:
                if s is None or s < 0:
                    parts.append(mx.random.normal((1, T, C)))
                else:
                    key = mx.random.key(int(s))
                    parts.append(mx.random.normal((1, T, C), key=key))
            return mx.concatenate(parts, axis=0).astype(dt)
        key = mx.random.key(int(_seed))
        return mx.random.normal((bsz, T, C), key=key).astype(dt)

    noise = _draw_noise(seed)
    # Retake mixing: variance-preserving blend with an independent noise draw.
    # v=0 -> noise unchanged; v=1 -> equivalent to using retake_seed as the main seed.
    if retake_variance > 0.0:
        retake_noise = _draw_noise(retake_seed)
        v_rad = retake_variance * (math.pi / 2.0)
        noise = math.cos(v_rad) * noise + math.sin(v_rad) * retake_noise

    # ---- Timestep schedule ----
    t_schedule_list = get_timestep_schedule(shift, timesteps, infer_steps=infer_steps)
    num_steps = len(t_schedule_list)

    cover_steps = int(num_steps * audio_cover_strength)

    # ---- Prepare decoder step (compiled or plain with KV cache) ----
    _compiled_step = None
    if compile_model:

        def _raw_step(xt, t, tr, enc, ctx):
            vt, _ = mlx_decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=tr,
                encoder_hidden_states=enc,
                context_latents=ctx,
                cache=None,
                use_cache=False,
            )
            return vt

        try:
            _compiled_step = mx.compile(_raw_step)
            logger.info("[MLX-DiT] Diffusion step compiled with mx.compile().")
        # mx.compile() is a pure optimisation and the uncompiled path is always
        # correct, so any failure must degrade rather than propagate. MLX does
        # not document which exception types compilation can raise.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[MLX-DiT] mx.compile() failed (%s); using uncompiled path.", exc
            )

    # Note: Heun solver requires two model evaluations per step with
    # different inputs, so we disable KV caching when using it.
    if use_heun:
        cache = None
    else:
        cache = MLXCrossAttentionCache() if _compiled_step is None else None

    xt = noise
    prev_vt = None  # for EMA smoothing

    def _model_eval(x_input, t_val, enc, ctx_in, step_cache):
        """Single model evaluation helper."""
        # Every scalar the loop broadcasts -- timesteps here, step sizes below --
        # is built at ``dt``. ``mx.full`` infers float32 from a Python float, and
        # one float32 operand promotes the bf16 tensor it touches, so an implicit
        # dtype here leaks float32 into the timestep MLP and from there, through
        # the AdaLN modulation, into every layer. Upstream sizes the same arrays
        # off ``context_latents.dtype``.
        t_arr = mx.full((x_input.shape[0],), t_val, dtype=dt)
        if _compiled_step is not None:
            return _compiled_step(x_input, t_arr, t_arr, enc, ctx_in), step_cache
        vt_out, step_cache = mlx_decoder(
            hidden_states=x_input,
            timestep=t_arr,
            timestep_r=t_arr,
            encoder_hidden_states=enc,
            context_latents=ctx_in,
            cache=step_cache,
            use_cache=(not do_cfg and not use_heun),
        )
        return vt_out, step_cache

    def _apply_predictor_cfg(vt_raw, current_t_val):
        """Guide the predictor evaluation: APG, advancing the momentum state."""
        if not do_cfg:
            return vt_raw
        pred_cond = vt_raw[:bsz]
        pred_uncond = vt_raw[bsz:]
        if cfg_interval_start <= current_t_val <= cfg_interval_end:
            return _mlx_apg_forward(
                pred_cond, pred_uncond, guidance_scale, momentum_state
            )
        return pred_cond

    def _apply_corrector_cfg(vt_raw, current_t_val):
        """Guide the Heun corrector evaluation: plain CFG, no state.

        Routing the corrector through APG would step the momentum recurrence
        twice per interval, so from the second step onwards the running
        average -- and every guided velocity derived from it -- diverges from
        the reference. Upstream splits the two for the same reason.
        """
        if not do_cfg:
            return vt_raw
        pred_cond = vt_raw[:bsz]
        pred_uncond = vt_raw[bsz:]
        if cfg_interval_start <= current_t_val <= cfg_interval_end:
            return _mlx_cfg_forward(pred_cond, pred_uncond, guidance_scale)
        return pred_cond

    def _apply_stabilisation(vt_guided, xt_current, prev_velocity):
        """Apply optional norm clamping and EMA smoothing."""
        # Velocity norm clamping -- prevents outlier predictions
        if use_norm_clamp:
            vt_norm = mx.sqrt((vt_guided * vt_guided).sum(axis=(1, 2), keepdims=True))
            xt_norm = (
                mx.sqrt((xt_current * xt_current).sum(axis=(1, 2), keepdims=True))
                + 1e-10
            )
            scale = mx.minimum(
                mx.ones_like(vt_norm),
                (velocity_norm_threshold * xt_norm) / (vt_norm + 1e-10),
            )
            vt_guided = vt_guided * scale

        # Velocity EMA smoothing -- stabilises denoising trajectory
        if use_ema and prev_velocity is not None:
            vt_guided = (
                1.0 - velocity_ema_factor
            ) * vt_guided + velocity_ema_factor * prev_velocity

        return vt_guided

    diff_start = time.time()
    _switched_to_non_cover = False

    # DCW -- opt-in per-band wavelet-domain correction (CVPR 2026).  On MLX,
    # `haar` runs natively; other wavelets bridge through pytorch_wavelets
    # for output parity with the CUDA/CPU PyTorch path.  See
    # `acestep.models.mlx.dcw_correction_mlx`.
    from .dcw import apply_mlx_dcw

    dcw_active = dcw_enabled and (
        dcw_scaler != 0.0 or (dcw_mode == "double" and dcw_high_scaler != 0.0)
    )
    if dcw_active:
        logger.info(
            "[MLX-DiT] DCW enabled (mode=%s, scaler=%.3f, high_scaler=%.3f, wavelet=haar).",
            dcw_mode,
            dcw_scaler,
            dcw_high_scaler,
        )

    # The schedule carries no trailing zero, so pair every timestep with its
    # successor and close the last interval at t=0. Upstream iterates
    # ``zip(t[:-1], t[1:])`` over a schedule that ends at zero, which gives the
    # final interval the same treatment as the rest; special-casing it as a bare
    # Euler hop to zero drops the corrector evaluation that Heun owes it.
    step_pairs = list(zip(t_schedule_list, [*t_schedule_list[1:], 0.0], strict=True))

    for step_idx, (current_t, next_t) in enumerate(
        tqdm(step_pairs, desc="MLX DiT diffusion", disable=disable_tqdm)
    ):
        # Switch to non-cover conditions when appropriate
        if step_idx >= cover_steps and not _switched_to_non_cover:
            _switched_to_non_cover = True
            if enc_hs_nc is not None:
                enc_hs = enc_hs_nc
                ctx = ctx_nc
            if cache is not None:
                cache = MLXCrossAttentionCache()

        # Build input: double batch for CFG
        x_in = mx.concatenate([xt, xt], axis=0) if do_cfg else xt

        # ---- First model evaluation (predictor) ----
        vt, cache = _model_eval(x_in, current_t, enc_hs, ctx, cache)
        mx.eval(vt)

        vt = _apply_predictor_cfg(vt, current_t)
        vt = _apply_stabilisation(vt, xt, prev_vt)

        # Cache pre-step latent so DCW can reconstruct the predicted clean
        # sample ``denoised = x_before - v * t`` after the sampler update.
        # Also stash the raw velocity (pre-Heun-averaging) so the x0
        # reconstruction uses the single-evaluation ``v(t_curr)``, matching
        # the reference FLUX scheduler's ``x0 = sample - sigma * v``.
        xt_before_step = xt
        vt_for_denoise = vt

        # Step size of this interval. Named ``delta``, not ``dt``: ``dt`` is the
        # compute dtype, and the SDE branch below casts with it.
        delta = current_t - next_t

        if use_heun:
            # ---- Heun (second-order) ODE step ----
            # Predictor: Euler step to get xt_predicted at next_t
            delta_arr = mx.full((bsz, 1, 1), delta, dtype=dt)
            xt_predicted = xt - vt * delta_arr
            mx.eval(xt_predicted)

            # Corrector: evaluate model at the predicted point
            x_in2 = (
                mx.concatenate([xt_predicted, xt_predicted], axis=0)
                if do_cfg
                else xt_predicted
            )
            vt2, cache = _model_eval(x_in2, next_t, enc_hs, ctx, cache)
            mx.eval(vt2)
            vt2 = _apply_corrector_cfg(vt2, next_t)
            vt2 = _apply_stabilisation(vt2, xt_predicted, vt)

            # Average the two velocity predictions (trapezoidal rule)
            vt_avg = 0.5 * (vt + vt2)
            xt = xt - vt_avg * delta_arr
            vt = vt_avg  # store averaged velocity for EMA
        elif infer_method == "sde":
            t_unsq = mx.full((bsz, 1, 1), current_t, dtype=dt)
            pred_clean = xt - vt * t_unsq
            # next_t == 0 on the last interval: the blend is then all clean
            # sample, so skip the draw rather than scale it away -- MLX's
            # implicit PRNG state would advance for nothing.
            if next_t > 0.0:
                new_noise = mx.random.normal(xt.shape).astype(dt)
                xt = next_t * new_noise + (1.0 - next_t) * pred_clean
            else:
                xt = pred_clean
        else:
            # ---- Standard Euler ODE step ----
            delta_arr = mx.full((bsz, 1, 1), delta, dtype=dt)
            xt = xt - vt * delta_arr

        mx.eval(xt)

        # DCW correction -- push x_next's frequency bands away from the
        # predicted clean sample.  Scaler decays with t_curr so this is
        # identity at t=0 and strongest at t≈1.
        if dcw_active:
            t_unsq_d = mx.full((bsz, 1, 1), current_t, dtype=dt)
            denoised = xt_before_step - vt_for_denoise * t_unsq_d
            xt = apply_mlx_dcw(
                xt,
                denoised,
                t_curr=current_t,
                enabled=True,
                mode=dcw_mode,
                scaler=dcw_scaler,
                high_scaler=dcw_high_scaler,
            )
            mx.eval(xt)

        prev_vt = vt  # store for EMA

        # ---- Repaint step injection ----
        if do_repaint:
            injection_cutoff = round(repaint_injection_ratio * num_steps)
            if step_idx < injection_cutoff:
                xt = _mlx_repaint_step_injection(
                    xt, clean_src_mx, repaint_mask_mx, next_t, noise
                )
                mx.eval(xt)

    # ---- Repaint boundary blend (post-loop) ----
    if do_repaint and repaint_crossfade_frames > 0:
        xt = _mlx_repaint_boundary_blend(
            xt, clean_src_mx, repaint_mask_np, repaint_crossfade_frames
        )
        mx.eval(xt)

    diff_end = time.time()
    total_end = time.time()

    time_costs["diffusion_time_cost"] = diff_end - diff_start
    time_costs["diffusion_per_step_time_cost"] = time_costs[
        "diffusion_time_cost"
    ] / max(num_steps, 1)
    time_costs["total_time_cost"] = total_end - total_start
    time_costs["sampler_mode"] = effective_sampler_mode

    # numpy has no bfloat16, and MLX refuses the buffer protocol for it, so the
    # loop's dtype has to be widened here rather than at the call site.
    result_np = np.array(xt.astype(mx.float32))
    return {
        "target_latents": result_np,
        "time_costs": time_costs,
    }
