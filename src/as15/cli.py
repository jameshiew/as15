"""Command line interface."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from .convert import PRECISIONS
from .models import (
    DEFAULT_MODEL,
    DEFAULT_PLANNER,
    MODELS,
    PLANNERS,
    ModelSpec,
    resolve,
)
from .pipeline import (
    MAX_DURATION,
    MIN_DURATION,
    OUTPUT_SUFFIX,
    GenerationRequest,
    check_output_path,
    generate,
    resolve_request,
    write_audio,
)

if TYPE_CHECKING:
    import numpy as np

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Generate music with ACE-Step 1.5 XL on Apple Silicon via MLX.",
)


def _err(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)


def _resolve_model(name: str) -> ModelSpec:
    """The checkpoint *name* names, as a usage error when it names none.

    ``resolve`` is a library helper and raises ValueError; turning it into a
    BadParameter here is what keeps an unknown ``--model`` looking like every
    other bad option rather than like a crash.
    """
    try:
        return resolve(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _channels(audio: np.ndarray) -> str:
    """How to describe the decoded audio's channel layout.

    Read off the array rather than asserted: this line used to say "stereo"
    unconditionally, while nothing between the VAE config and here requires
    the checkpoint to have two output channels.
    """
    count = audio.shape[1] if audio.ndim > 1 else 1
    return {1: "mono", 2: "stereo"}.get(count, f"{count} channels")


def _read_codes(path: Path | None) -> tuple[int, ...] | None:
    """The plan in *path*, as a usage error when there is not one.

    Read here rather than in the pipeline so a mistyped path or a file that
    holds no plan costs a second, rather than surfacing after the checkpoints
    have downloaded.
    """
    if path is None:
        return None
    from .codes import read_codes

    try:
        return read_codes(path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _read_lyrics(lyrics: str | None, lyrics_file: Path | None) -> str:
    if lyrics_file is not None:
        if str(lyrics_file) == "-":
            return sys.stdin.read()
        if not lyrics_file.is_file():
            raise typer.BadParameter(f"No such file: {lyrics_file}")
        return lyrics_file.read_text()
    return lyrics or ""


@app.command()
def sing(
    prompt: Annotated[
        str,
        typer.Option(
            "--prompt",
            "-p",
            help="Style prompt, e.g. 'dream pop, female vocals, warm analog tape'.",
        ),
    ],
    lyrics: Annotated[
        str | None, typer.Option("--lyrics", "-l", help="Lyrics as a string.")
    ] = None,
    lyrics_file: Annotated[
        Path | None,
        typer.Option(
            "--lyrics-file",
            "-L",
            help="Read lyrics from a file, or '-' for stdin.",
        ),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help=f"Output file; must end in {OUTPUT_SUFFIX}."),
    ] = Path("song.flac"),
    model: Annotated[
        str, typer.Option("--model", "-m", help=f"One of: {', '.join(MODELS)}.")
    ] = DEFAULT_MODEL,
    duration: Annotated[
        float,
        typer.Option(
            "--duration",
            "-d",
            min=MIN_DURATION,
            max=MAX_DURATION,
            help="Seconds.",
        ),
    ] = 120.0,
    steps: Annotated[
        int | None, typer.Option("--steps", "-s", min=1, help="Diffusion steps.")
    ] = None,
    guidance: Annotated[
        float | None,
        typer.Option(
            "--guidance",
            "-g",
            help="CFG scale; 1.0 turns it off, below that is rejected. "
            "Ignored by turbo.",
        ),
    ] = None,
    shift: Annotated[
        float | None,
        typer.Option("--shift", help="Timestep shift. Default depends on model."),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Random seed. Omit for random.")
    ] = None,
    language: Annotated[
        str, typer.Option("--language", help="Vocal language code.")
    ] = "en",
    bpm: Annotated[int | None, typer.Option("--bpm", help="Target tempo.")] = None,
    key: Annotated[
        str | None, typer.Option("--key", help="Key/scale, e.g. 'C major'.")
    ] = None,
    time_signature: Annotated[
        int | None, typer.Option("--time-signature", help="2, 3, 4 or 6.")
    ] = None,
    sampler: Annotated[str, typer.Option("--sampler", help="euler or heun.")] = "euler",
    dcw: Annotated[
        bool | None,
        typer.Option(
            "--dcw/--no-dcw",
            help=(
                "Wavelet-domain correction. Default: on for turbo, off for "
                "sft/base, where it causes mushy, distorted output."
            ),
        ),
    ] = None,
    precision: Annotated[
        str, typer.Option("--precision", help=f"One of: {', '.join(PRECISIONS)}.")
    ] = "bf16",
    plan: Annotated[
        bool,
        typer.Option(
            "--plan/--no-plan",
            help=(
                "Sketch the song with the 5Hz planner LM first, and condition "
                "on that instead of on silence. Slower, and downloads a second "
                "checkpoint."
            ),
        ),
    ] = False,
    planner: Annotated[
        str,
        typer.Option(
            "--planner", help=f"Planner for --plan. One of: {', '.join(PLANNERS)}."
        ),
    ] = DEFAULT_PLANNER,
    planner_seed: Annotated[
        int | None,
        typer.Option(
            "--planner-seed",
            help="Seed for --plan. Defaults to --seed; pin it to keep one plan "
            "while varying the render.",
        ),
    ] = None,
    audio_codes: Annotated[
        Path | None,
        typer.Option(
            "--audio-codes",
            help="Condition on the plan in this file, as written by `as15 plan`.",
        ),
    ] = None,
    device: Annotated[
        str, typer.Option("--device", help="Torch device for conditioning.")
    ] = "auto",
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            # Only the diffusion bar. A cold cache still reports its download
            # (huggingface_hub's own bars) and its conversion, both of which
            # are minutes of apparent silence otherwise.
            help="No diffusion progress bar.",
        ),
    ] = False,
) -> None:
    """Generate a song from a style prompt and lyrics."""
    spec = _resolve_model(model)
    lyrics_text = _read_lyrics(lyrics, lyrics_file)

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    if plan and audio_codes is not None:
        raise typer.BadParameter(
            "--plan writes a plan and --audio-codes supplies one; pass one or "
            "the other."
        )
    codes = _read_codes(audio_codes)

    request = GenerationRequest(
        style_prompt=prompt,
        lyrics=lyrics_text,
        duration=duration,
        language=language,
        bpm=bpm,
        key_scale=key,
        time_signature=time_signature,
        steps=steps,
        guidance=guidance,
        shift=shift,
        seed=seed,
        sampler=sampler,
        dcw=dcw,
        precision=precision,
        audio_codes=codes,
        planner=planner if plan else None,
        # One --seed reproduces the whole run, plan included. Pinning
        # --planner-seed on its own keeps the plan while --seed moves the
        # render, which is how you hear what the diffusion is contributing.
        planner_seed=planner_seed if planner_seed is not None else seed,
    )

    # The same call generate() makes, so the banner below cannot report
    # settings other than the ones that run -- and so a bad option costs a
    # second rather than the minutes it takes to reach the diffusion loop.
    # Same for --out: what the write will insist on, asked before the run
    # rather than after it.
    try:
        resolved = resolve_request(spec, request)
        check_output_path(out)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None

    # Overwriting is the policy -- rerunning a command should not need the
    # last take cleared out of the way first -- but it is said out loud here,
    # while there is still time to stop, rather than discovered afterwards.
    if out.exists():
        typer.secho(
            f"{out} exists and will be overwritten.", fg=typer.colors.YELLOW, err=True
        )

    if not lyrics_text.strip():
        typer.secho(
            "No lyrics given - generating an instrumental. "
            "Pass --lyrics or --lyrics-file for vocals.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    typer.secho(f"model     {spec.key}  ({spec.repo_id}@{spec.revision[:8]})", err=True)
    # The resolved duration rather than the one typed: the latent grid is 40 ms,
    # so `-d 12.9` is a 12.92 s take, and that is the length the metas block,
    # the decode and the file's own AS15_DURATION all say too.
    typer.secho(
        f"duration  {resolved.duration:g}s   steps {resolved.steps}   "
        f"seed {resolved.seed}",
        err=True,
    )
    typer.secho(
        f"sampling  shift {resolved.shift:g}   guidance {resolved.guidance:g}   "
        f"dcw {'on' if resolved.dcw else 'off'}",
        err=True,
    )
    if resolved.planner is not None:
        chosen = PLANNERS[resolved.planner]
        typer.secho(
            f"planning  {chosen.key} ({chosen.gigabytes:g} GB)   "
            f"seed {resolved.planner_seed}",
            err=True,
        )
    elif resolved.audio_codes is not None:
        typer.secho(f"plan      {len(resolved.audio_codes)} codes given", err=True)

    # Imported here rather than at the top of the module: as15.conditioning
    # pulls in torch, and `as15 models` should not pay for that.
    from .conditioning import InputTooLong

    # Only the tokenizer can tell whether the prompt and lyrics fit, and it
    # loads with the conditioner, so this is the one bad-input case that
    # survives past the banner. Reported as an error rather than raised as a
    # usage error, which would print the whole help text underneath the
    # settings we just listed.
    try:
        result = generate(spec, request, device=device, progress=not quiet)
    except InputTooLong as exc:
        _err(str(exc))
        raise typer.Exit(2) from None

    # A generation this far in is worth more than a traceback: an unwritable
    # destination or a decode that came back non-finite is reported as an
    # error, having left whatever was at *out* alone.
    try:
        write_audio(out, result.audio, result.sample_rate, result.tags)
    except (ValueError, OSError) as exc:
        _err(f"Could not write {out}: {exc}")
        raise typer.Exit(1) from None

    t = result.timings
    seconds = len(result.audio) / result.sample_rate
    typer.secho(
        f"\nwrote {out}  ({seconds:.1f}s audio, {result.sample_rate} Hz "
        f"{_channels(result.audio)})",
        fg=typer.colors.GREEN,
        err=True,
    )
    typer.secho(
        "  diffusion {:.1f}s  decode {:.1f}s  peak {:.1f} GB".format(
            t.get("diffusion_time_cost", 0.0),
            t.get("decode", 0.0),
            t.get("peak_memory_gb", 0.0),
        ),
        err=True,
    )


@app.command("models")
def list_models() -> None:
    """List available checkpoints."""
    for key, spec in MODELS.items():
        default = "  (default)" if key == DEFAULT_MODEL else ""
        typer.echo(
            f"{key}{default}\n  {spec.description}\n"
            f"  {spec.repo_id}@{spec.revision[:8]}\n"
        )
    typer.echo("5Hz planners (--plan --planner KEY):\n")
    for key, planner in PLANNERS.items():
        default = "  (default)" if key == DEFAULT_PLANNER else ""
        location = f"{planner.repo_id}@{planner.revision[:8]}"
        if planner.subdir:
            location += f"  {planner.subdir}"
        typer.echo(
            f"{key}{default}\n  {planner.description}\n"
            f"  {location}  ({planner.gigabytes:g} GB)\n"
        )


@app.command()
def plan(
    prompt: Annotated[
        str, typer.Option("--prompt", "-p", help="Style prompt, as for `sing`.")
    ],
    lyrics: Annotated[
        str | None, typer.Option("--lyrics", "-l", help="Lyrics as a string.")
    ] = None,
    lyrics_file: Annotated[
        Path | None,
        typer.Option("--lyrics-file", "-L", help="Read lyrics from a file, or '-'."),
    ] = None,
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Where to write the plan.")
    ] = Path("plan.codes"),
    duration: Annotated[
        float,
        typer.Option("--duration", "-d", min=MIN_DURATION, max=MAX_DURATION),
    ] = 120.0,
    planner: Annotated[
        str, typer.Option("--planner", help=f"One of: {', '.join(PLANNERS)}.")
    ] = DEFAULT_PLANNER,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Random seed. Omit for random.")
    ] = None,
    language: Annotated[str, typer.Option("--language")] = "en",
    bpm: Annotated[int | None, typer.Option("--bpm")] = None,
    key: Annotated[str | None, typer.Option("--key")] = None,
    time_signature: Annotated[int | None, typer.Option("--time-signature")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Write an audio-code plan without rendering it.

    The same plan `sing --plan` writes, kept as a file so it can be rendered
    more than once: at different guidance, with a different sampler, or against
    the other checkpoint. Planning a two-minute song is one pass of the LM;
    rendering it is fifty passes of a 4 B DiT, so separating them is what makes
    trying six renders of one plan affordable.
    """
    from .models import latent_frames_for, resolve_planner
    from .pipeline import planner_path

    try:
        spec = resolve_planner(planner)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None

    request = GenerationRequest(
        style_prompt=prompt,
        lyrics=_read_lyrics(lyrics, lyrics_file),
        duration=duration,
        language=language,
        bpm=bpm,
        key_scale=key,
        time_signature=time_signature,
        seed=seed,
    )
    try:
        resolved = resolve_request(resolve(DEFAULT_MODEL), request)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    from .codes import format_codes
    from .planner import write_plan

    typer.secho(f"planner   {spec.key} ({spec.gigabytes:g} GB)   seed {seed}", err=True)
    written = write_plan(
        planner_path(spec),
        resolved,
        latent_frames_for(duration),
        seed,
        progress=not quiet,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    from .atomic import publish

    publish(
        out,
        lambda tmp: tmp.write_text(format_codes(written.codes), encoding="utf-8"),
    )
    typer.secho(f"\n{written.reasoning}", err=True)
    typer.secho(
        f"wrote {out}  ({len(written.codes)} codes, {resolved.duration:g}s)",
        fg=typer.colors.GREEN,
        err=True,
    )


@app.command()
def download(
    model: Annotated[
        str, typer.Option("--model", "-m", help=f"One of: {', '.join(MODELS)}.")
    ] = DEFAULT_MODEL,
    precision: Annotated[
        str, typer.Option("--precision", help=f"One of: {', '.join(PRECISIONS)}.")
    ] = "bf16",
) -> None:
    """Fetch a checkpoint and pre-build its MLX weight cache."""
    from .convert import convert_dit, convert_vae
    from .pipeline import _resolve_snapshots

    if precision not in PRECISIONS:
        raise typer.BadParameter(f"--precision must be one of: {', '.join(PRECISIONS)}")

    spec = _resolve_model(model)
    dit_snapshot, base_snapshot = _resolve_snapshots(spec)
    dit_path = convert_dit(dit_snapshot, precision)
    vae_path = convert_vae(base_snapshot)
    typer.secho(
        f"DiT  {dit_path}  ({dit_path.stat().st_size / 1e9:.2f} GB)",
        fg=typer.colors.GREEN,
    )
    typer.secho(
        f"VAE  {vae_path}  ({vae_path.stat().st_size / 1e9:.2f} GB)",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
