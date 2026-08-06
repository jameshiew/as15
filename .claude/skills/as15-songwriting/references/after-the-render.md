# After the render

Read this once a take is worth keeping, or when someone asks for an edit as15
does not do.

## What as15 will not do

The model has more tasks than this CLI exposes. There is **no remix, cover,
repaint, extend or reference-audio path here.** as15 implements the
text-to-audio half only, and the audio-to-codes tokenizer the editing tasks need
is never even built (`src/as15/conditioning.py`). Advice found elsewhere about
Remix strength, Repaint windows or a reference track is describing upstream's
Gradio and ComfyUI surfaces, not this one, and there is no flag here to map it
onto.

There is also **no negative prompt.** CFG's unconditional branch is a stored
null embedding rather than text you supply, so "no guitars" is just the word
*guitars* in your caption. The lever is a short positive palette, said twice in
different words.

What there is instead: re-render the same plan at another seed, another
guidance, or the other checkpoint. That gives you variations of one song rather
than a transformation of a recording -- and for a section that will not come out
right, it is regenerate-and-comp rather than repair in place.

## Finish it outside the generator

The complaints that recur about ACE-Step output are vocal expressiveness,
dynamic contrast, drum transitions, groove and how memorable the melody is.
Those are the hardest things to generate and none of them is a step count. as15
writes 48 kHz stereo FLAC, which is a fine thing to import:

- comp the best sections together -- takes rendered from one plan share an
  arrangement, so they are the ones most likely to line up, but check both
  boundaries by ear;
- automate the verse-to-chorus level change the model under-plays;
- reinforce or replace the drums, particularly transitions and fills;
- de-ess, tame the harshness, clear the low end;
- master last, once the arrangement has stopped moving.

Takes come back with their peak pinned near the ceiling (0.998993, the same
value across every seed -- a pipeline ceiling rather than four takes that each
happened to peak), so there is no headroom to work into. Pull the level down
before anything else.

## Recovering the recipe

Every take carries its own recipe in Vorbis comments -- see **What a take
remembers** in the README for the full field list.

```bash
ffprobe -v error -show_entries format_tags -of default out/final.flac
```

The style prompt comes back as `comment`: ffmpeg maps the Vorbis `DESCRIPTION`
field onto its own key, so asking for `format_tags=DESCRIPTION` returns nothing.
Every `AS15_*` field keeps its name, so a single value pulls out cleanly --
which is how you recover the seed of a take worth re-rendering:

```bash
ffprobe -v error -show_entries format_tags=AS15_SEED -of csv=p=0 out/final.flac
```

## Originality

MIT covers this code and upstream's weights. It says nothing about anyone else's
songs. Upstream's own guidance is to check output for originality, disclose that
a track is AI-generated, and get permission for protected material used as a
source or a reference -- which in practice means keeping artist names out of
`-p` and existing lyrics out of the sheet.
