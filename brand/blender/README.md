# M2NWU caustic film (Blender)

`m2nwu_caustic.py` builds the M2NWU caustic scene from an empty file and renders it:
sunlight passes through a shaped glass slab and focuses the wordmark **M2NWU** onto an
ink floor as light, with red/green/blue fringing. Over a seamless 4 s loop the glass
tilts, breathes and ripples, so the light smears and fringes, then refocuses.

## How it works

1. **Mask.** The wordmark is set in a Blender text object, converted to triangles and
   rasterised in numpy. No extra Python packages are needed.
2. **Lens height field.** A jump-flood distance transform finds each stroke's medial
   line and local half-width `w`. The profile `h = (1-k)(w d - d^2/2) / R` turns every
   stroke into a cylindrical converging lens. `R = (n-1) * floor distance`, so each
   stroke focuses into a band `k` (`FOCUS_RATIO`) times its width, right on the floor.
3. **Glass.** That field displaces a real glass mesh (grid + Solidify). A geometry-nodes
   modifier adds the ripple waves. Tilt, breathing (Z scale = focus pull) and ripple
   amplitude/phase are keyed on every frame.
4. **Caustic.** Each frame, the same surface, tilt, scale and ripple are forward
   ray-traced with Snell's law (curved top, flat bottom). This runs three times, once per
   spectral band (`IOR_RGB`), and writes an HDR EXR caustic sequence. The ink floor
   emits it through a tone curve: focused light reads white-hot, and the unfocused field
   is a dim cobalt wash. This gives clean caustics at preview sample counts, which
   path-traced refractive caustics on a CPU cannot. The glass casts no shadow, because
   its light is the caustic.
5. **Render.** Cycles CPU with OpenImageDenoise, AgX view transform and compositor
   bloom (or `--post-bloom`, see below). The script writes PNG frames and assembles
   an MP4 with ffmpeg.

All motion is a function of loop phase (`sin^2` envelopes, integer wave harmonics), so
frame N+1 equals frame 1. Frame 1 is the focus moment.

## Run

Headless with Blender:

    blender -b -P m2nwu_caustic.py -- --out OUT --frames 96              # final 1920x1080
    blender -b -P m2nwu_caustic.py -- --out OUT --frames 48 --preview    # 960x540 preview

With the pip `bpy` module (`pip install bpy`, Python 3.13 for bpy 5.3):

    python m2nwu_caustic.py --out OUT --frames 48 --preview --post-bloom

In the Blender UI, open the Scripting tab, open the file and click Run Script. This
builds the scene and saves the .blend but does not render, unless `RENDER_IN_UI = True`.
After that, render with F12 / Ctrl+F12 as usual. The caustic EXR sequence stays in
`OUT/caustic_<shot>/`, and the .blend refers to it.

Flags:

| flag | meaning |
|---|---|
| `--out DIR` | output folder (.blend, caustic EXRs, frames, mp4) |
| `--frames N` | loop length in frames (default 96 = 4 s at 24 fps) |
| `--preview` / `--quality final` | quality preset (see `QUALITY`) |
| `--shot wordmark\|emblem` | the wordmark, or the M2 emblem (ring + "M2") |
| `--camera hero\|top` | low 3/4 hero camera, or orthographic top-down camera |
| `--stills 1,13,25` | render only these frames (as PNG) |
| `--samples N` | override the sample count |
| `--font PATH` | font file to use (Syne ExtraBold recommended) |
| `--post-bloom` | bloom in numpy instead of the compositor (needed with no GPU/EGL context) |
| `--no-render` | build and save the .blend only |

Main parameters at the top of the script: `TEXT`, `INK`, `COBALT`, `FPS`, `LOOP_SECONDS`,
glass size/thickness/height (`GLASS_*`, `HP`), `IOR_RGB` (wider spread = stronger
fringes), `FOCUS_RATIO`, `TILT_DEG`, `BREATH`, `RIPPLE_AMP`, `WAVES`, tone curve
(`TONE_KNEE`, `TONE_POWER`, `EMISSION`, `FIELD_GLOW`), `SUN_DIR`, `FIELD_ALTERNATE`
(keys the floor colour ink > cobalt > ink) and `THERMAL` (ink > cobalt > cyan > lime >
white ramp).

## Font

The brand face is Syne ExtraBold. If the script finds `Syne-ExtraBold.ttf` next to it
(or in `fonts/`), it uses that file. Otherwise it falls back to Inter ExtraBold and then
to Blender's built-in font. The committed preview was rendered with **Inter ExtraBold**,
because Syne could not be downloaded offline.

## Render-time notes (4 CPU cores, no GPU)

- Preview (960x540, 16 samples + denoise): about 6.5 s per frame. Tracing the caustic
  takes about 1.5 s per frame. 48 frames take about 7 minutes in total.
- Final (1920x1080, 128 samples, 3072x1728 caustic, 3x3 rays per texel): expect roughly
  15x the preview time per frame on the same CPU. A GPU cuts the Cycles part a lot.
- The bpy wheel cannot create a GPU context on a headless server, so the compositor
  aborts there. Use `--post-bloom` in that case. Desktop Blender uses the compositor's
  Bloom glare.
- Path-traced caustics are switched off on purpose (`caustics_refractive = False`),
  because the floor already emits the traced caustic. To compare against Cycles MNEE
  shadow caustics, enable caustic caster/receiver flags and use a spot light instead.
