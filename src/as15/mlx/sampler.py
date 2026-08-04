# MLX diffusion generation loop for AceStep DiT decoder.
#
# Replicates the timestep scheduling and ODE/SDE stepping from
# ``AceStepConditionGenerationModel.generate_audio`` using pure MLX arrays.
#
# Sampler modes (see issue #957):
# - ``euler``: First-order Euler ODE/SDE step (default, original behaviour).
# - ``heun``: Second-order Heun predictor-corrector -- evaluates the model
#   twice per step and averages the predictions for higher accuracy, which
#   matters especially with 8-step turbo inference.
#
# Upstream's loop carries a good deal more than that: cover switching, repaint,
# retake, ``mx.compile``, an explicit timestep list, velocity norm clamping and
# velocity EMA. None of it is reachable from this CLI or from
# ``GenerationRequest``, and none of it checked its arguments -- so the way to
# find out that a repaint mask was the wrong shape, that a cover switch had been
# given only half its conditioning, or that a timestep list had collapsed onto a
# duplicate and was stepping backwards, was to listen to the output. All of it
# is gone. Restoring one means restoring its argument checks with it.

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


def check_schedule_options(shift: float, infer_steps: int) -> None:
    """Reject step counts and shifts the schedule cannot express.

    Both used to be accepted and then quietly mean something else. A step count
    below 1 fell through to a fixed 8-step lookup table, so the CLI reported the
    number it was asked for while the loop ran eight steps; with that table gone
    it would instead build an empty schedule and hand back the initial noise as
    if it were a song. And shift=0 divided 0 by 0 at t=1 (the map's denominator
    is ``1+(shift-1)*t``), while a negative shift either hit that same zero
    denominator part-way down the schedule or produced a non-monotonic one.
    Nothing downstream can tell those apart from a deliberate setting.
    """
    if infer_steps < 1:
        raise ValueError(f"infer_steps must be at least 1, got {infer_steps}.")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError(f"shift must be finite and greater than zero, got {shift}.")


def check_sampling_options(
    sampler_mode: str,
    infer_method: str,
    shift: float,
    infer_steps: int,
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


def check_guidance(guidance: float) -> None:
    """Reject a guidance scale that does not mean what it says.

    CFG only engages above 1.0, so 0.5 and -10 run the identical
    conditional-only pass 1.0 gives while being reported back as the value the
    caller asked for, and inf enters the guidance arithmetic and takes the
    latents non-finite with it. Shared with the pipeline, which applies it to a
    request before anything loads.
    """
    if not math.isfinite(guidance) or guidance < 1.0:
        raise ValueError(
            f"guidance must be finite and at least 1.0, where 1.0 means no CFG; "
            f"got {guidance}."
        )


def get_timestep_schedule(shift: float, infer_steps: int) -> list[float]:
    """Timesteps for one run: a descending linspace warped by *shift*.

    ``t_i = 1 - i/infer_steps``, mapped through ``shift*t / (1+(shift-1)*t)``,
    matching the PyTorch base model. The trailing zero is left off; the loop
    closes the last interval at t=0 itself.

    Upstream also accepts an explicit timestep list, which it snaps entry by
    entry onto a 20-value table without checking that the result still
    descends -- two nearby entries collapse onto one value and produce a
    zero-length interval, an ascending list produces negative ones -- and falls
    back to a fixed 8-step table when neither a list nor a step count is given.
    Nothing here passes either, so both are gone.

    Raises:
        ValueError: If ``infer_steps`` is below 1, or ``shift`` is not a finite
            positive number.
    """
    check_schedule_options(shift, infer_steps)

    schedule = [1.0 - i / infer_steps for i in range(infer_steps)]
    if shift != 1.0:
        schedule = [shift * t / (1.0 + (shift - 1.0) * t) for t in schedule]
    return schedule


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


def mlx_generate_diffusion(
    mlx_decoder,
    encoder_hidden_states_np: np.ndarray,
    context_latents_np: np.ndarray,
    src_latents_shape: tuple[int, ...],
    infer_steps: int,
    seed: int | None = None,
    infer_method: str = "ode",
    shift: float = 3.0,
    guidance_scale: float = 1.0,
    null_condition_emb_np: np.ndarray | None = None,
    sampler_mode: str = "euler",
    dcw_enabled: bool = True,
    compute_dtype: str = "bfloat16",
    disable_tqdm: bool = False,
) -> DiffusionResult:
    """Run the complete MLX diffusion loop with optional CFG guidance.

    Numpy at the boundary in both directions, and only at the boundary: the
    conditioning that comes in is the one stage that still runs under torch,
    and the latents that go out are handed straight back to the MLX VAE.
    Upstream passes torch tensors here and converts on both sides.

    Args:
        mlx_decoder: ``MLXDiTDecoder`` instance with loaded weights.
        encoder_hidden_states_np: [B, enc_L, D] from ``Conditioner.build``.
        context_latents_np: [B, T, C] from ``Conditioner.build``.
        src_latents_shape: shape tuple [B, T, 64] for noise generation.
        infer_steps: number of diffusion steps.
        seed: random seed, or None to draw unseeded.
        infer_method: "ode" or "sde".
        shift: timestep shift factor.
        guidance_scale: CFG guidance strength; 1.0 runs unguided.
        null_condition_emb_np: [1, 1, D] null condition embedding, required
            whenever ``guidance_scale`` is above 1.0.
        sampler_mode: Sampler algorithm -- ``"euler"`` (first-order, default) or
            ``"heun"`` (second-order predictor-corrector for cleaner output).
        dcw_enabled: Apply the per-step wavelet-domain correction.
        compute_dtype: MLX dtype the whole loop runs in.
        disable_tqdm: If True, suppress the diffusion progress bar.

    Returns:
        Dict with ``"target_latents"`` (numpy) and ``"time_costs"`` dict.
        ``time_costs["sampler_mode"]`` names the sampler that ran, which is
        ``"euler"`` when a Heun request degraded under SDE.

    Raises:
        ValueError: If ``sampler_mode``, ``infer_method``, ``infer_steps``,
            ``shift`` or ``guidance_scale`` is outside what this loop supports.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)

    from .dit import MLXCrossAttentionCache

    check_sampling_options(sampler_mode, infer_method, shift, infer_steps)
    check_guidance(guidance_scale)
    # Guiding needs something to guide against. Asking for CFG without a null
    # embedding used to fall through to the ordinary conditional pass, so a
    # distilled checkpoint -- which ships no null branch -- ran unguided while
    # the caller was told it was guiding at 7.0.
    if guidance_scale > 1.0 and null_condition_emb_np is None:
        raise ValueError(
            f"guidance_scale={guidance_scale} needs a null_condition_emb_np to "
            f"guide against; pass one, or guidance_scale=1.0 to run without CFG."
        )

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

    if use_heun:
        logger.info(
            "[MLX-DiT] Using Heun (second-order) sampler for higher-quality output."
        )

    time_costs: dict[str, float | str] = {}
    total_start = time.time()

    # Run the loop in the same dtype as the decoder weights. Feeding fp32
    # activations to bf16 weights makes MLX promote the weights per-op, which
    # costs both bandwidth and peak memory on a 4B model.
    dt = getattr(mx, compute_dtype)

    enc_hs = mx.array(encoder_hidden_states_np).astype(dt)
    ctx = mx.array(context_latents_np).astype(dt)

    bsz = src_latents_shape[0]
    T = src_latents_shape[1]
    C = src_latents_shape[2]

    # ---- CFG setup ----
    do_cfg = guidance_scale > 1.0
    if do_cfg:
        null_cond = mx.array(null_condition_emb_np).astype(dt)
        null_expanded = mx.broadcast_to(null_cond, enc_hs.shape)
        enc_hs = mx.concatenate([enc_hs, null_expanded], axis=0)
        ctx = mx.concatenate([ctx, ctx], axis=0)
    momentum_state: dict | None = {} if do_cfg else None

    # ---- Noise preparation ----
    # ``None`` marks an unseeded draw, which stays on MLX's implicit global
    # stream -- there is no reproducibility to preserve for one.
    base_key = mx.random.key(int(seed)) if seed is not None else None

    def _draw_noise(key):
        """Draw [B, T, C] noise from *key*, or unseeded when it is None."""
        if key is None:
            return mx.random.normal((bsz, T, C)).astype(dt)
        return mx.random.normal((bsz, T, C), key=key).astype(dt)

    noise = _draw_noise(base_key)

    # ---- Timestep schedule ----
    t_schedule_list = get_timestep_schedule(shift, infer_steps)
    num_steps = len(t_schedule_list)

    # Keys for the fresh noise every SDE interval draws. Those draws used to go
    # through MLX's implicit global PRNG state, so the request seed fixed only
    # the initial noise: any other ``mx.random`` call in the process -- an
    # earlier generation, a caller's own draw between two requests -- moved the
    # whole SDE trajectory. Split an explicit stream off the seed instead. The
    # initial draw above still uses the base key itself, so its values, and with
    # them the entire ODE path, are unchanged.
    step_keys: list = []
    if infer_method == "sde":
        step_keys = (
            [None] * num_steps
            if base_key is None
            else list(mx.random.split(base_key, num_steps))
        )

    # The cache holds cross-attention K/V, which are projected from
    # ``encoder_hidden_states`` and nothing else -- no noisy latent, no
    # timestep. One entry per layer therefore stays valid across every step,
    # across both of Heun's evaluations, and across CFG's doubled batch, whose
    # conditioning is doubled the same way; nothing in the loop changes the
    # conditioning. Caching was previously off for CFG and for Heun -- that is,
    # on the whole default SFT path -- which reprojected the conditioning
    # through every layer's k_proj and v_proj on all 100 evaluations of a
    # 50-step run.
    cache = MLXCrossAttentionCache()

    xt = noise

    def _model_eval(x_input, t_val, step_cache):
        """Single model evaluation helper."""
        # Every scalar the loop broadcasts -- timesteps here, step sizes below --
        # is built at ``dt``. ``mx.full`` infers float32 from a Python float, and
        # one float32 operand promotes the bf16 tensor it touches, so an implicit
        # dtype here leaks float32 into the timestep MLP and from there, through
        # the AdaLN modulation, into every layer. Upstream sizes the same arrays
        # off ``context_latents.dtype``.
        t_arr = mx.full((x_input.shape[0],), t_val, dtype=dt)
        return mlx_decoder(
            hidden_states=x_input,
            timestep=t_arr,
            timestep_r=t_arr,
            encoder_hidden_states=enc_hs,
            context_latents=ctx,
            cache=step_cache,
            use_cache=True,
        )

    def _apply_predictor_cfg(vt_raw):
        """Guide the predictor evaluation: APG, advancing the momentum state."""
        if not do_cfg:
            return vt_raw
        return _mlx_apg_forward(
            vt_raw[:bsz], vt_raw[bsz:], guidance_scale, momentum_state
        )

    def _apply_corrector_cfg(vt_raw):
        """Guide the Heun corrector evaluation: plain CFG, no state.

        Routing the corrector through APG would step the momentum recurrence
        twice per interval, so from the second step onwards the running
        average -- and every guided velocity derived from it -- diverges from
        the reference. Upstream splits the two for the same reason.
        """
        if not do_cfg:
            return vt_raw
        return _mlx_cfg_forward(vt_raw[:bsz], vt_raw[bsz:], guidance_scale)

    diff_start = time.time()

    # DCW -- per-band wavelet-domain correction (CVPR 2026), on by default for
    # the distilled checkpoints and off for the rest.  See `.dcw`.
    from .dcw import apply_mlx_dcw

    if dcw_enabled:
        logger.info("[MLX-DiT] DCW enabled (double-band Haar correction).")

    # The schedule carries no trailing zero, so pair every timestep with its
    # successor and close the last interval at t=0. Upstream iterates
    # ``zip(t[:-1], t[1:])`` over a schedule that ends at zero, which gives the
    # final interval the same treatment as the rest; special-casing it as a bare
    # Euler hop to zero drops the corrector evaluation that Heun owes it.
    step_pairs = list(zip(t_schedule_list, [*t_schedule_list[1:], 0.0], strict=True))

    for step_idx, (current_t, next_t) in enumerate(
        tqdm(step_pairs, desc="MLX DiT diffusion", disable=disable_tqdm)
    ):
        # Build input: double batch for CFG
        x_in = mx.concatenate([xt, xt], axis=0) if do_cfg else xt

        # ---- First model evaluation (predictor) ----
        vt, cache = _model_eval(x_in, current_t, cache)
        mx.eval(vt)

        vt = _apply_predictor_cfg(vt)

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
            vt2, cache = _model_eval(x_in2, next_t, cache)
            mx.eval(vt2)
            vt2 = _apply_corrector_cfg(vt2)

            # Average the two velocity predictions (trapezoidal rule)
            vt_avg = 0.5 * (vt + vt2)
            xt = xt - vt_avg * delta_arr
        elif infer_method == "sde":
            t_unsq = mx.full((bsz, 1, 1), current_t, dtype=dt)
            pred_clean = xt - vt * t_unsq
            # next_t == 0 on the last interval: the blend is then all clean
            # sample, so skip the draw rather than scale it away.
            if next_t > 0.0:
                new_noise = _draw_noise(step_keys[step_idx])
                xt = next_t * new_noise + (1.0 - next_t) * pred_clean
            else:
                xt = pred_clean
        else:
            # ---- Standard Euler ODE step ----
            delta_arr = mx.full((bsz, 1, 1), delta, dtype=dt)
            xt = xt - vt * delta_arr

        mx.eval(xt)

        # DCW correction -- push x_next's frequency bands away from the
        # predicted clean sample, the coarse band hardest at the top of the
        # schedule and the detail band hardest as it lands.
        if dcw_enabled:
            t_unsq_d = mx.full((bsz, 1, 1), current_t, dtype=dt)
            denoised = xt_before_step - vt_for_denoise * t_unsq_d
            xt = apply_mlx_dcw(xt, denoised, t_curr=current_t)
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
