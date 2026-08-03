"""Command line interface."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Annotated

import typer

from .models import DEFAULT_MODEL, MODELS, resolve
from .pipeline import GenerationRequest, generate, write_audio

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Generate music with ACE-Step 1.5 XL on Apple Silicon via MLX.",
)


def _err(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)


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
    out: Annotated[Path, typer.Option("--out", "-o", help="Output audio file.")] = Path(
        "song.wav"
    ),
    model: Annotated[
        str, typer.Option("--model", "-m", help=f"One of: {', '.join(MODELS)}.")
    ] = DEFAULT_MODEL,
    duration: Annotated[
        float, typer.Option("--duration", "-d", min=10, max=600, help="Seconds.")
    ] = 120.0,
    steps: Annotated[
        int | None, typer.Option("--steps", "-s", help="Diffusion steps.")
    ] = None,
    guidance: Annotated[
        float | None,
        typer.Option("--guidance", "-g", help="CFG scale (ignored by turbo)."),
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
        str, typer.Option("--precision", help="bf16 or fp32.")
    ] = "bf16",
    device: Annotated[
        str, typer.Option("--device", help="Torch device for conditioning.")
    ] = "auto",
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="No progress bar.")
    ] = False,
) -> None:
    """Generate a song from a style prompt and lyrics."""
    spec = resolve(model)

    if sampler not in {"euler", "heun"}:
        raise typer.BadParameter("--sampler must be 'euler' or 'heun'")
    if precision not in {"bf16", "fp32"}:
        raise typer.BadParameter("--precision must be 'bf16' or 'fp32'")

    lyrics_text = _read_lyrics(lyrics, lyrics_file)
    if not lyrics_text.strip():
        typer.secho(
            "No lyrics given - generating an instrumental. "
            "Pass --lyrics or --lyrics-file for vocals.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

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
    )

    eff_steps = steps if steps is not None else spec.steps
    eff_shift = shift if shift is not None else spec.shift
    eff_dcw = dcw if dcw is not None else spec.dcw
    eff_guidance = guidance if guidance is not None else spec.guidance
    if not spec.supports_cfg:
        eff_guidance = 1.0
    typer.secho(f"model     {spec.key}  ({spec.repo_id})", err=True)
    typer.secho(f"duration  {duration:g}s   steps {eff_steps}   seed {seed}", err=True)
    typer.secho(
        f"sampling  shift {eff_shift:g}   guidance {eff_guidance:g}   "
        f"dcw {'on' if eff_dcw else 'off'}",
        err=True,
    )

    result = generate(spec, request, device=device, progress=not quiet)
    write_audio(out, result.audio, result.sample_rate)

    t = result.timings
    seconds = len(result.audio) / result.sample_rate
    typer.secho(
        f"\nwrote {out}  ({seconds:.1f}s audio, {result.sample_rate} Hz stereo)",
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
        typer.echo(f"{key}{default}\n  {spec.description}\n  {spec.repo_id}\n")


@app.command()
def download(
    model: Annotated[
        str, typer.Option("--model", "-m", help=f"One of: {', '.join(MODELS)}.")
    ] = DEFAULT_MODEL,
    precision: Annotated[str, typer.Option("--precision")] = "bf16",
) -> None:
    """Fetch a checkpoint and pre-build its MLX weight cache."""
    from .convert import convert_dit, convert_vae
    from .pipeline import _resolve_snapshots

    spec = resolve(model)
    dit_snapshot, base_snapshot = _resolve_snapshots(spec)
    dit_path = convert_dit(dit_snapshot, spec.cache_name, precision)
    vae_path = convert_vae(base_snapshot / "vae")
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
