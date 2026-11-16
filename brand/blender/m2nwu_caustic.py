"""M2NWU - caustic brand film, built from scratch in Blender (bpy).

Sunlight refracts through a shaped glass slab and focuses the wordmark "M2NWU"
(or the M2 emblem) onto an ink floor, with chromatic dispersion. The glass tilts,
breathes and ripples, so the light smears, fringes and refocuses in a seamless loop.

How the caustic is made
  1. The wordmark is rasterised from a Blender text object (no external libraries).
  2. A distance transform of the mask turns every stroke into a cylindrical
     converging lens whose focal length equals the glass-to-floor distance, so at the
     focus moment each stroke collapses into a bright line of light on the floor.
  3. That height field displaces the real glass mesh in the scene, and the same field
     is forward ray-traced (Snell's law, top surface + flat bottom, three spectral
     bands with different IOR) per frame into an HDR caustic texture that the floor
     emits. This gives clean, noise-free caustics at preview sample counts, which
     path-traced refractive caustics on a CPU cannot.
  Glass tilt, breathing (focus) and ripples are keyed per frame on the glass object
  and its geometry-nodes modifier, and the tracer uses exactly the same values.

Run
  blender -b -P m2nwu_caustic.py -- --out DIR --frames 48 --preview
  python  m2nwu_caustic.py --out DIR --frames 48 --preview      (pip "bpy" module)
  In the Blender UI: Scripting tab > open this file > Run Script (builds the scene,
  does not render unless RENDER_IN_UI is True).
"""

import math
import os
import shutil
import subprocess
import sys
import time

import bpy
import numpy as np

# ---------------------------------------------------------------- parameters
TEXT = "M2NWU"
EMBLEM_TEXT = "M2"
FONT_CANDIDATES = [            # Syne ExtraBold first; bold geometric fallbacks after
    "Syne-ExtraBold.ttf", "fonts/Syne-ExtraBold.ttf",
    "/usr/share/fonts/truetype/syne/Syne-ExtraBold.ttf",
    "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf",
    "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
INK = "#05070f"
COBALT = "#1c2bff"
FPS = 24
LOOP_SECONDS = 4.0             # loop length; --frames overrides the frame count
FIELD_ALTERNATE = False        # key the floor colour ink -> cobalt -> ink over the loop
THERMAL = False                # thermal ramp variant (ink > cobalt > cyan > lime > white)

# optics (metres)
GLASS_W, GLASS_H = 7.6, 2.6    # glass slab footprint
GLASS_T = 0.06                 # slab thickness
HP = 2.2                       # glass bottom to floor
IOR_RGB = (1.472, 1.522, 1.572)  # spectral bands; widen for stronger fringing
FOCUS_RATIO = 0.34             # each stroke focuses into a band this fraction of its width
TEXT_HEIGHT = 1.45             # cap height of the wordmark on the glass
FLOOR_W, FLOOR_H = 10.4, 5.85  # caustic texture domain on the floor (16:9)

# motion (all periodic in the loop phase -> seamless)
TILT_DEG = 3.5                 # max glass tilt
BREATH = 0.35                  # height-field scale swing (focus pull)
RIPPLE_AMP = 0.0085            # ripple amplitude in metres
WAVES = [  # (kx, ky, amplitude weight, temporal harmonic, phase)
    (3.1, 1.2, 1.00, 1, 0.0), (-1.7, 3.6, 0.80, 2, 1.3),
    (5.3, -2.2, 0.45, 3, 2.1), (-4.6, -4.1, 0.35, 2, 4.0), (8.0, 0.9, 0.25, 4, 0.7),
]

# look
TONE_KNEE = 1.08               # caustic density below this stays dark (flat field = 1.0)
TONE_POWER = 1.6               # tone curve above the knee
EMISSION = 2.2                 # emission of focused light
FIELD_GLOW = 0.025             # dim cobalt wash of the unfocused field
SUN_DIR = (0.10, 0.06)         # sun direction tilt (x, y) from vertical; keeps its glint off the top view
BLOOM = True

QUALITY = {
    "preview": dict(res=(960, 540), samples=16, tex=(1536, 864), ss=2, grid=(640, 220)),
    "final": dict(res=(1920, 1080), samples=128, tex=(3072, 1728), ss=3, grid=(1280, 440)),
}
RENDER_IN_UI = False


# ---------------------------------------------------------------- helpers
def hex_rgb(h):
    h = h.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    a = dict(out=os.path.abspath("m2nwu_out"), frames=int(round(FPS * LOOP_SECONDS)),
             quality="final", shot="wordmark", camera="hero", render=True,
             stills=None, samples=None, font=None, post_bloom=False)
    it = iter(argv)
    for k in it:
        if k == "--out": a["out"] = os.path.abspath(next(it))
        elif k == "--frames": a["frames"] = int(next(it))
        elif k == "--preview": a["quality"] = "preview"
        elif k == "--quality": a["quality"] = next(it)
        elif k == "--shot": a["shot"] = next(it)            # wordmark | emblem
        elif k == "--camera": a["camera"] = next(it)        # hero | top
        elif k == "--samples": a["samples"] = int(next(it))
        elif k == "--stills": a["stills"] = [int(x) for x in next(it).split(",")]
        elif k == "--font": a["font"] = next(it)
        elif k == "--no-render": a["render"] = False
        elif k == "--post-bloom": a["post_bloom"] = True   # no GPU/EGL: bloom in numpy instead of compositor
    return a


def find_font(user):
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for p in ([user] if user else []) + FONT_CANDIDATES:
        for q in (p, os.path.join(here, p)):
            if q and os.path.isfile(q):
                return q
    return None  # Blender's built-in font


def shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    H, W = a.shape[:2]
    if abs(dy) >= H or abs(dx) >= W:
        return out
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


def edt_inside(mask):
    """Euclidean distance (px) from each inside pixel to the nearest outside pixel (jump flooding)."""
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    seed = ~mask
    sy = np.where(seed, yy, -1e9).astype(np.float32)
    sx = np.where(seed, xx, -1e9).astype(np.float32)
    step = 1 << int(math.ceil(math.log2(max(H, W)))) - 1
    passes = []
    while step >= 1:
        passes.append(step); step //= 2
    passes += [1]
    for s in passes:
        best = (sy - yy) ** 2 + (sx - xx) ** 2
        for dy in (-s, 0, s):
            for dx in (-s, 0, s):
                if dy == 0 and dx == 0:
                    continue
                cy, cx = shift(sy, dy, dx, -1e9), shift(sx, dy, dx, -1e9)
                d = (cy - yy) ** 2 + (cx - xx) ** 2
                m = d < best
                sy[m], sx[m], best[m] = cy[m], cx[m], d[m]
    return np.where(mask, np.sqrt(best), 0.0).astype(np.float32), sy, sx


def max_filter(a, r):
    out = a.copy()
    s = 1
    while s <= r:            # doubling window, roughly square of radius ~r
        out = np.maximum(out, np.maximum(shift(out, s, 0, 0), shift(out, -s, 0, 0)))
        out = np.maximum(out, np.maximum(shift(out, 0, s, 0), shift(out, 0, -s, 0)))
        s *= 2
    return out


def blur(a, sigma):
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    a = sum(k[i] * p[i:i + a.shape[0]] for i in range(2 * r + 1))
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    return sum(k[i] * p[:, i:i + a.shape[1]] for i in range(2 * r + 1))


# ---------------------------------------------------------------- mask from Blender text
def text_triangles(body, font_path, size):
    cu = bpy.data.curves.new("mask_text", "FONT")
    cu.body = body
    if font_path:
        cu.font = bpy.data.fonts.load(font_path, check_existing=True)
    cu.align_x, cu.align_y, cu.size = "CENTER", "CENTER", size
    ob = bpy.data.objects.new("mask_text", cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    me = ob.evaluated_get(dg).to_mesh()
    me.calc_loop_triangles()
    co = np.array([v.co[:2] for v in me.vertices], np.float32)
    tri = np.array([t.vertices[:] for t in me.loop_triangles], np.int64)
    pts = co[tri]
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob); bpy.data.curves.remove(cu)
    return pts  # (n,3,2)


def raster_tris(pts, x0, y0, cs, W, H, aa=2):
    S = aa
    m = np.zeros((H * S, W * S), np.float32)
    c = cs / S
    for t in pts:
        (ax, ay), (bx, by), (cx, cy) = t
        i0 = max(int((min(ax, bx, cx) - x0) / c), 0); i1 = min(int((max(ax, bx, cx) - x0) / c) + 2, W * S)
        j0 = max(int((min(ay, by, cy) - y0) / c), 0); j1 = min(int((max(ay, by, cy) - y0) / c) + 2, H * S)
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(x0 + (np.arange(i0, i1) + 0.5) * c, y0 + (np.arange(j0, j1) + 0.5) * c)
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-12:
            continue
        l1 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / d
        l2 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / d
        inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        m[j0:j1, i0:i1] = np.maximum(m[j0:j1, i0:i1], inside)
    return m.reshape(H, S, W, S).mean(axis=(1, 3))


def build_mask(shot, font, cs, W, H):
    x0, y0 = -GLASS_W / 2, -GLASS_H / 2
    if shot == "emblem":
        yy, xx = np.mgrid[0:H, 0:W]
        X, Y = x0 + (xx + .5) * cs, y0 + (yy + .5) * cs
        r = np.hypot(X, Y)
        R = GLASS_H * 0.46
        ring = ((r > R * 0.86) & (r < R)).astype(np.float32)
        disc = ((r > R * 0.40) & (r < R * 0.70)).astype(np.float32) * 0  # disc reads via ring + glyphs
        pts = text_triangles(EMBLEM_TEXT, font, 1.0)
        span = pts[..., 1].max() - pts[..., 1].min()
        pts = pts * (R * 0.78 / span)
        glyph = raster_tris(pts, x0, y0, cs, W, H)
        return np.clip(ring + disc + glyph, 0, 1)
    pts = text_triangles(TEXT, font, 1.0)
    capspan = pts[..., 1].max() - pts[..., 1].min()
    k = TEXT_HEIGHT / capspan
    wspan = (pts[..., 0].max() - pts[..., 0].min()) * k
    k *= min(1.0, GLASS_W * 0.9 / wspan)
    pts = pts * k
    pts[..., 1] -= (pts[..., 1].max() + pts[..., 1].min()) / 2
    return raster_tris(pts, x0, y0, cs, W, H)


def lens_heightfield(mask, cs):
    """Each stroke becomes a cylindrical lens focusing on its medial line at the floor.

    For every inside pixel we march away from its nearest edge until the distance field
    stops growing: that is the stroke's medial line, and its distance w is the local
    half-width. The parabolic profile h = (w d - d^2/2) / R has slope (w - d)/R, which
    bends each ray onto the medial line exactly at the floor (thin-lens, f = R/(n-1)).
    """
    inside = mask > 0.5
    d, sy, sx = edt_inside(inside)
    H, W = d.shape
    yy, xx = np.nonzero(inside)
    dp = d[yy, xx]
    uy, ux = yy - sy[yy, xx], xx - sx[yy, xx]
    nn = np.maximum(np.hypot(uy, ux), 1e-6); uy, ux = uy / nn, ux / nn
    run = np.zeros_like(dp); alive = np.ones(dp.shape, bool)
    for st in range(1, int(d.max()) + 2):
        qy = np.clip(np.rint(yy + uy * st).astype(np.int64), 0, H - 1)
        qx = np.clip(np.rint(xx + ux * st).astype(np.int64), 0, W - 1)
        ok = alive & (d[qy, qx] >= dp + st * 0.80)
        run[ok] = st; alive &= ok
        if not alive.any():
            break
    w = np.zeros_like(d); w[yy, xx] = dp + run * 0.9
    # smooth w inside the strokes (normalised blur) so the profile stays integrable
    ws = blur(w, 3.0) / np.maximum(blur(inside.astype(np.float32), 3.0), 1e-3)
    w = np.where(inside, np.maximum(ws, d), 0)
    dd = np.minimum(d, w)
    n = IOR_RGB[1]
    Rc = (n - 1.0) * (HP + GLASS_T / n)          # lens radius so f = floor distance
    h = (1 - FOCUS_RATIO) * (w * dd - 0.5 * dd * dd) * cs * cs / Rc   # slope (1-k)(w-d)cs/Rc
    h = blur(h, 0.8)
    return h.astype(np.float32)


# ---------------------------------------------------------------- motion (shared by tracer + keys)
def motion(phase):
    """phase in [0,1). Focus moment at phase 0. Everything periodic."""
    env = math.sin(math.pi * phase) ** 2                     # 0 at focus, 1 mid-loop
    a = 2 * math.pi * phase
    tilt_x = math.radians(TILT_DEG) * env * math.sin(a + 0.6)
    tilt_y = math.radians(TILT_DEG) * env * math.cos(a)
    scale_z = 1.0 + BREATH * math.sin(a) * (0.5 + 0.5 * env)  # focus pull through the loop
    ripple = RIPPLE_AMP * env ** 0.8
    return dict(tilt_x=tilt_x, tilt_y=tilt_y, scale_z=scale_z, ripple=ripple, phi=a)


def ripple_grad(X, Y, m):
    gx = np.zeros_like(X); gy = np.zeros_like(X)
    for kx, ky, w, harm, ph in WAVES:
        c = np.cos(kx * X + ky * Y + harm * m["phi"] + ph) * (w * m["ripple"])
        gx += c * kx; gy += c * ky
    return gx, gy


def refract(d, n, eta):
    """d incident (unit, ...,3), n normal facing against d, eta = n1/n2."""
    cosi = -(d * n).sum(-1, keepdims=True)
    k = 1 - eta * eta * (1 - cosi * cosi)
    k = np.maximum(k, 0)
    return eta * d + (eta * cosi - np.sqrt(k)) * n


def trace_caustic(h, cs, m, tex):
    TW, TH = tex
    tx = FLOOR_W / TW
    gy0, gx0 = np.gradient(h, cs)
    H, W = h.shape
    X, Y = np.meshgrid(-GLASS_W / 2 + (np.arange(W) + .5) * cs, -GLASS_H / 2 + (np.arange(H) + .5) * cs)
    rgx, rgy = ripple_grad(X, Y, m)
    gx = gx0 * m["scale_z"] + rgx
    gy = gy0 * m["scale_z"] + rgy
    nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    # glass rotation (X then Y), world light straight down
    cx, sx, cy, sy = math.cos(m["tilt_x"]), math.sin(m["tilt_x"]), math.cos(m["tilt_y"]), math.sin(m["tilt_y"])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rm = Ry @ Rx
    sd = np.array([SUN_DIR[0], SUN_DIR[1], -1.0]); sd /= np.linalg.norm(sd)
    d_local = Rm.T @ sd
    d = np.broadcast_to(d_local, nrm.shape)
    zc = HP + GLASS_T                         # glass top height (object origin)
    out = np.zeros((TH, TW, 3), np.float32)
    wgt = (cs / tx) ** 2
    for b, ior in enumerate(IOR_RGB):
        t1 = refract(d, nrm, 1.0 / ior)
        t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
        s = GLASS_T / np.maximum(-t1[..., 2], 1e-3)
        px = X + t1[..., 0] * s; py = Y + t1[..., 1] * s
        t2 = refract(t1, np.array([0, 0, 1.0]), ior)
        # to world
        P = np.stack([px, py, np.full_like(px, -GLASS_T)], -1) @ Rm.T
        P[..., 2] += zc
        D = t2 @ Rm.T
        tt = P[..., 2] / np.maximum(-D[..., 2], 1e-3)
        fx = P[..., 0] + D[..., 0] * tt; fy = P[..., 1] + D[..., 1] * tt
        u = (fx + FLOOR_W / 2) / tx - 0.5; v = (fy + FLOOR_H / 2) / tx - 0.5
        i0 = np.floor(u).astype(np.int64); j0 = np.floor(v).astype(np.int64)
        fu = (u - i0).astype(np.float32); fv = (v - j0).astype(np.float32)
        acc = np.zeros(TW * TH, np.float64)
        for di, dj, ww in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
            ii, jj = i0 + di, j0 + dj
            ok = (ii >= 0) & (ii < TW) & (jj >= 0) & (jj < TH)
            acc += np.bincount((jj[ok] * TW + ii[ok]), weights=ww[ok] * wgt, minlength=TW * TH)
        out[..., b] = acc.reshape(TH, TW)
    # direct sun outside the glass footprint (soft edge)
    U, V = np.meshgrid(-FLOOR_W / 2 + (np.arange(TW) + .5) * tx, -FLOOR_H / 2 + (np.arange(TH) + .5) * tx)
    U = U - zc * sd[0] / -sd[2]; V = V - zc * sd[1] / -sd[2]     # glass shadow footprint
    ex = np.clip((np.abs(U) - GLASS_W / 2) / (2 * tx) + .5, 0, 1)
    ey = np.clip((np.abs(V) - GLASS_H / 2) / (2 * tx) + .5, 0, 1)
    out += np.maximum(ex, ey)[..., None]
    return out


def save_exr(arr, path):
    H, W = arr.shape[:2]
    img = bpy.data.images.new("tmp_exr", W, H, alpha=True, float_buffer=True)
    rgba = np.concatenate([arr, np.ones((H, W, 1), np.float32)], -1).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "OPEN_EXR"
    img.save()
    bpy.data.images.remove(img)


# ---------------------------------------------------------------- scene
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def make_glass(h, cs, grid):
    gw, gh = grid
    H, W = h.shape
    xs = np.linspace(-GLASS_W / 2, GLASS_W / 2, gw)
    ys = np.linspace(-GLASS_H / 2, GLASS_H / 2, gh)
    X, Y = np.meshgrid(xs, ys)
    ii = np.clip(((X + GLASS_W / 2) / cs - .5).round().astype(int), 0, W - 1)
    jj = np.clip(((Y + GLASS_H / 2) / cs - .5).round().astype(int), 0, H - 1)
    Z = h[jj, ii]
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    idx = np.arange(gw * gh).reshape(gh, gw)
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    me = bpy.data.meshes.new("glass")
    me.vertices.add(len(verts)); me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(q.size); me.loops.foreach_set("vertex_index", q.ravel())
    me.polygons.add(len(q)); me.polygons.foreach_set("loop_start", np.arange(0, q.size, 4))
    me.update(); me.validate()
    me.shade_smooth()
    ob = bpy.data.objects.new("Glass", me)
    ob.location = (0, 0, HP + GLASS_T)
    bpy.context.scene.collection.objects.link(ob)
    # ripple geometry nodes (same waves as the tracer)
    ng = bpy.data.node_groups.new("Ripple", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    vamp = N.new("ShaderNodeValue"); vamp.name = vamp.label = "Amplitude"   # keyed per frame
    vph = N.new("ShaderNodeValue"); vph.name = vph.label = "Phase"           # keyed per frame
    pos = N.new("GeometryNodeInputPosition")
    sep = N.new("ShaderNodeSeparateXYZ"); L.new(pos.outputs[0], sep.inputs[0])
    total = None
    for kx, ky, w, harm, ph in WAVES:
        ax = N.new("ShaderNodeMath"); ax.operation = "MULTIPLY"; ax.inputs[1].default_value = kx
        L.new(sep.outputs[0], ax.inputs[0])
        ay = N.new("ShaderNodeMath"); ay.operation = "MULTIPLY_ADD"; ay.inputs[1].default_value = ky
        L.new(sep.outputs[1], ay.inputs[0]); L.new(ax.outputs[0], ay.inputs[2])
        at = N.new("ShaderNodeMath"); at.operation = "MULTIPLY_ADD"; at.inputs[1].default_value = harm
        L.new(vph.outputs[0], at.inputs[0]); L.new(ay.outputs[0], at.inputs[2])
        ad = N.new("ShaderNodeMath"); ad.operation = "ADD"; ad.inputs[1].default_value = ph
        L.new(at.outputs[0], ad.inputs[0])
        sn = N.new("ShaderNodeMath"); sn.operation = "SINE"; L.new(ad.outputs[0], sn.inputs[0])
        mw = N.new("ShaderNodeMath"); mw.operation = "MULTIPLY"; mw.inputs[1].default_value = w
        L.new(sn.outputs[0], mw.inputs[0])
        if total is None:
            total = mw
        else:
            s = N.new("ShaderNodeMath"); s.operation = "ADD"
            L.new(total.outputs[0], s.inputs[0]); L.new(mw.outputs[0], s.inputs[1]); total = s
    amp = N.new("ShaderNodeMath"); amp.operation = "MULTIPLY"
    L.new(total.outputs[0], amp.inputs[0]); L.new(vamp.outputs[0], amp.inputs[1])
    comb = N.new("ShaderNodeCombineXYZ"); L.new(amp.outputs[0], comb.inputs[2])
    sp = N.new("GeometryNodeSetPosition")
    L.new(gi.outputs["Geometry"], sp.inputs["Geometry"]); L.new(comb.outputs[0], sp.inputs["Offset"])
    L.new(sp.outputs[0], go.inputs[0])
    mod = ob.modifiers.new("Ripple", "NODES"); mod.node_group = ng
    sol = ob.modifiers.new("Solidify", "SOLIDIFY"); sol.thickness = GLASS_T; sol.offset = -1; sol.use_even_offset = True
    # material: dispersive glass
    mat = bpy.data.materials.new("Glass")
    nt = mat.node_tree; nt.nodes.clear()
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bs.inputs["Base Color"].default_value = (0.96, 0.98, 1.0, 1)
    bs.inputs["Roughness"].default_value = 0.0
    bs.inputs["IOR"].default_value = IOR_RGB[1]
    bs.inputs["Transmission Weight"].default_value = 1.0
    try:
        bs.inputs["Transmission Dispersion Scale"].default_value = 1.0
        bs.inputs["Transmission Dispersion Abbe Number"].default_value = 18.0
    except KeyError:
        pass
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(bs.outputs[0], o.inputs[0])
    me.materials.append(mat)
    ob.visible_shadow = False          # its light is the floor caustic
    return ob, mod


def make_floor(seq_first, nframes):
    bpy.ops.mesh.primitive_plane_add(size=1)
    fl = bpy.context.active_object; fl.name = "Floor"
    fl.scale = (FLOOR_W * 3, FLOOR_H * 3, 1)
    mat = bpy.data.materials.new("FloorCaustic"); nt = mat.node_tree; nt.nodes.clear()
    N, L = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")      # object coords -> caustic UV
    mp.inputs["Scale"].default_value = (3.0 * FLOOR_W / FLOOR_W, 3.0, 1)
    mp.inputs["Location"].default_value = (0.5, 0.5, 0)
    mp.inputs["Scale"].default_value = (fl.scale[0] / FLOOR_W, fl.scale[1] / FLOOR_H, 1)
    L.new(tc.outputs["Object"], mp.inputs[0])
    tex = N.new("ShaderNodeTexImage"); tex.extension = "EXTEND"; tex.interpolation = "Cubic"
    img = bpy.data.images.load(seq_first); img.source = "SEQUENCE"
    img.colorspace_settings.name = "Linear Rec.709" if "Linear Rec.709" in [c.name for c in bpy.types.ColorManagedInputColorspaceSettings.bl_rna.properties["name"].enum_items] else img.colorspace_settings.name
    tex.image = img
    tu = tex.image_user; tu.frame_duration = nframes; tu.frame_start = 1; tu.frame_offset = 0
    tu.use_auto_refresh = True; tu.use_cyclic = True
    L.new(mp.outputs[0], tex.inputs[0])
    # tone: hot = EMISSION * max(c - TONE_KNEE, 0)^TONE_POWER per channel (keeps the fringes),
    #       glow = c * FIELD_GLOW * cobalt, so the flat field reads as a dim cobalt wash
    sub = N.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"; sub.inputs[1].default_value = (TONE_KNEE,) * 3
    L.new(tex.outputs[0], sub.inputs[0])
    mx = N.new("ShaderNodeVectorMath"); mx.operation = "MAXIMUM"; mx.inputs[1].default_value = (0, 0, 0)
    L.new(sub.outputs[0], mx.inputs[0])
    pw = N.new("ShaderNodeVectorMath"); pw.operation = "POWER"; pw.inputs[1].default_value = (TONE_POWER,) * 3
    L.new(mx.outputs[0], pw.inputs[0])
    hot = N.new("ShaderNodeVectorMath"); hot.operation = "SCALE"; hot.inputs["Scale"].default_value = EMISSION
    L.new(pw.outputs[0], hot.inputs[0])
    gl = N.new("ShaderNodeVectorMath"); gl.operation = "MULTIPLY"
    gl.inputs[1].default_value = tuple(c * FIELD_GLOW for c in hex_rgb(COBALT))
    L.new(tex.outputs[0], gl.inputs[0])
    sm = N.new("ShaderNodeVectorMath"); sm.operation = "ADD"
    L.new(hot.outputs[0], sm.inputs[0]); L.new(gl.outputs[0], sm.inputs[1])
    col = sm.outputs[0]
    if THERMAL:
        bw = N.new("ShaderNodeRGBToBW"); L.new(col, bw.inputs[0])
        mr = N.new("ShaderNodeMath"); mr.operation = "MULTIPLY"; mr.inputs[1].default_value = 0.12
        L.new(bw.outputs[0], mr.inputs[0])
        ramp = N.new("ShaderNodeValToRGB"); L.new(mr.outputs[0], ramp.inputs[0])
        cr = ramp.color_ramp
        stops = [(0.0, INK), (0.25, COBALT), (0.5, "#19e6ff"), (0.75, "#b6ff2e"), (1.0, "#ffffff")]
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for e, (p, c) in zip(cr.elements, stops):
            e.position = p; e.color = (*hex_rgb(c), 1)
        sc = N.new("ShaderNodeVectorMath"); sc.operation = "SCALE"; sc.inputs["Scale"].default_value = 6.0
        L.new(ramp.outputs[0], sc.inputs[0]); col = sc.outputs[0]
    em = N.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = EMISSION
    L.new(col, em.inputs["Color"])
    df = N.new("ShaderNodeBsdfDiffuse"); df.inputs["Color"].default_value = (*hex_rgb(INK), 1)
    add = N.new("ShaderNodeAddShader"); L.new(df.outputs[0], add.inputs[0]); L.new(em.outputs[0], add.inputs[1])
    out = N.new("ShaderNodeOutputMaterial"); L.new(add.outputs[0], out.inputs[0])
    fl.data.materials.append(mat)
    if FIELD_ALTERNATE:
        c = df.inputs["Color"]
        for f, hx in ((1, INK), (nframes // 2 + 1, COBALT), (nframes + 1, INK)):
            c.default_value = (*hex_rgb(hx), 1); c.keyframe_insert("default_value", frame=f)
    return fl


def make_lights_cameras(scene):
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 1.5; sun.data.angle = math.radians(0.53)
    sd = np.array([SUN_DIR[0], SUN_DIR[1], -1.0])
    from mathutils import Vector
    sun.rotation_euler = Vector(sd).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(sun)
    rim = bpy.data.objects.new("CobaltRim", bpy.data.lights.new("CobaltRim", "AREA"))
    rim.data.energy = 120; rim.data.shape = "RECTANGLE"; rim.data.size = 14; rim.data.size_y = 0.25; rim.data.color = hex_rgb(COBALT)
    rim.location = (0, 7.5, 3.2); rim.rotation_euler = (math.radians(-75), 0, 0)
    scene.collection.objects.link(rim)
    key = bpy.data.objects.new("SoftKey", bpy.data.lights.new("SoftKey", "AREA"))
    key.data.energy = 250; key.data.size = 8
    key.location = (-6, -2, 6); key.rotation_euler = (math.radians(40), math.radians(-40), 0)
    key.visible_diffuse = False         # only glints on the glass
    scene.collection.objects.link(key)

    def cam(name, loc, target, lens=None, ortho=None):
        c = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        c.location = loc
        d = np.array(target) - np.array(loc)
        c.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]), 0, math.atan2(d[1], d[0]) - math.pi / 2)
        if ortho:
            c.data.type = "ORTHO"; c.data.ortho_scale = ortho
        else:
            c.data.lens = lens
        c.data.clip_end = 200
        scene.collection.objects.link(c)
        return c
    hero = cam("CamHero", (0.0, -6.3, 6.4), (0.0, -0.25, 0.0), lens=30)
    top = cam("CamTop", (0, 0, 14), (0, 0, 0), ortho=FLOOR_W * 0.92)
    top.rotation_euler = (0, 0, 0)
    return hero, top


def setup_render(scene, q, args, out):
    scene.render.engine = "CYCLES"
    cy = scene.cycles
    cy.device = "CPU"
    cy.samples = args["samples"] or q["samples"]
    cy.use_adaptive_sampling = True
    cy.use_denoising = True
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    cy.max_bounces = 8; cy.transmission_bounces = 8; cy.glossy_bounces = 3
    cy.diffuse_bounces = 1; cy.caustics_reflective = False; cy.caustics_refractive = False
    cy.blur_glossy = 1.0
    scene.render.resolution_x, scene.render.resolution_y = q["res"]
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.use_persistent_data = True
    vs = scene.view_settings
    try:
        vs.view_transform = "AgX"; vs.look = "AgX - Punchy"
    except TypeError:
        pass
    vs.exposure = 0.0
    world = bpy.data.worlds.new("Ink"); scene.world = world
    wn = world.node_tree; bg = wn.nodes.get("Background") or wn.nodes.new("ShaderNodeBackground")
    if not wn.nodes.get("World Output"):
        wo = wn.nodes.new("ShaderNodeOutputWorld"); wn.links.new(bg.outputs[0], wo.inputs[0])
    bg.inputs["Color"].default_value = (*hex_rgb(INK), 1); bg.inputs["Strength"].default_value = 1.0
    if BLOOM and not args["post_bloom"]:
        try:
            ng = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
            ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
            rl = ng.nodes.new("CompositorNodeRLayers")
            gl = ng.nodes.new("CompositorNodeGlare")
            gl.inputs["Type"].default_value = "Bloom"
            gl.inputs["Threshold"].default_value = 0.9
            gl.inputs["Strength"].default_value = 0.55
            gl.inputs["Size"].default_value = 0.6
            go = ng.nodes.new("NodeGroupOutput")
            ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
            ng.links.new(gl.outputs[0], go.inputs[0])
            scene.compositing_node_group = ng
            scene.render.use_compositing = True
        except Exception as e:  # compositor API differs between versions
            print("bloom disabled:", e)


def key_motion(glass, mod, nframes):
    va = mod.node_group.nodes["Amplitude"].outputs[0]
    vp = mod.node_group.nodes["Phase"].outputs[0]
    for f in range(1, nframes + 2):
        m = motion(((f - 1) % nframes) / nframes)
        glass.rotation_euler = (m["tilt_x"], m["tilt_y"], 0)
        glass.scale = (1, 1, m["scale_z"])
        glass.keyframe_insert("rotation_euler", frame=f)
        glass.keyframe_insert("scale", frame=f)
        va.default_value = m["ripple"]; va.keyframe_insert("default_value", frame=f)
        vp.default_value = m["phi"] if f <= nframes else 2 * math.pi
        vp.keyframe_insert("default_value", frame=f)
    ad = glass.animation_data
    try:
        for fc in ad.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"
    except AttributeError:  # layered actions (4.4+): interpolation default is fine per-frame keys
        pass


def post_bloom(path, strength=0.6, thresh=0.62):
    """Compositor-free bloom for machines without a GPU context (headless servers)."""
    img = bpy.data.images.load(path)
    W, H = img.size
    a = np.empty(W * H * 4, np.float32); img.pixels.foreach_get(a)
    a = a.reshape(H, W, 4)
    lin = a[..., :3] ** 2.2
    hi = np.maximum(lin - thresh ** 2.2, 0)
    small = hi[::4, ::4]
    glow = np.stack([blur(small[..., c], 2.5) + 0.6 * blur(small[..., c], 9.0) for c in range(3)], -1)
    glow = np.repeat(np.repeat(glow, 4, 0), 4, 1)[:H, :W]
    out = np.clip(lin + strength * glow, 0, 1) ** (1 / 2.2)
    a[..., :3] = out
    img.pixels.foreach_set(a.ravel())
    img.filepath_raw = path; img.file_format = "PNG"; img.save()
    bpy.data.images.remove(img)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    q = QUALITY[args["quality"]]
    out = args["out"]; os.makedirs(out, exist_ok=True)
    nfr = args["frames"]
    t0 = time.time()
    scene = reset_scene()
    font = find_font(args["font"])
    print("font:", font or "Blender built-in (Syne ExtraBold not found)")

    TW, TH = q["tex"]
    cs = FLOOR_W / TW / q["ss"]                     # ray spacing on the glass
    GW, GH = int(GLASS_W / cs), int(GLASS_H / cs)
    mask = build_mask(args["shot"], font, cs, GW, GH)
    h = lens_heightfield(mask, cs)
    print(f"height field {GW}x{GH}, max {h.max()*1000:.2f} mm, {time.time()-t0:.1f}s")

    cdir = os.path.join(out, f"caustic_{args['shot']}"); os.makedirs(cdir, exist_ok=True)
    frames_needed = range(1, nfr + 1) if not args["stills"] else sorted(set(args["stills"]))
    for f in frames_needed:
        c = trace_caustic(h, cs, motion((f - 1) / nfr), (TW, TH))
        save_exr(c, os.path.join(cdir, f"c_{f:04d}.exr"))
    if args["stills"]:   # the sequence node needs every frame on disk; reuse nearest traced frame
        have = sorted(set(args["stills"]))
        for f in range(1, nfr + 1):
            p = os.path.join(cdir, f"c_{f:04d}.exr")
            if not os.path.exists(p):
                src = min(have, key=lambda x: abs(x - f))
                shutil.copy(os.path.join(cdir, f"c_{src:04d}.exr"), p)
    print(f"caustics traced: {time.time()-t0:.1f}s")

    glass, mod = make_glass(h, cs, q["grid"])
    make_floor(os.path.join(cdir, "c_0001.exr"), nfr)
    hero, top = make_lights_cameras(scene)
    scene.camera = top if args["camera"] == "top" else hero
    setup_render(scene, q, args, out)
    scene.frame_start, scene.frame_end = 1, nfr
    key_motion(glass, mod, nfr)
    scene.frame_set(1)
    blend = os.path.join(out, f"m2nwu_caustic_{args['shot']}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    print("saved", blend)

    if not args["render"] or (bpy.app.background is False and not RENDER_IN_UI):
        return
    fdir = os.path.join(out, f"frames_{args['shot']}_{args['camera']}"); os.makedirs(fdir, exist_ok=True)
    tr = time.time()
    if args["stills"]:
        for f in args["stills"]:
            scene.frame_set(f)
            scene.render.filepath = os.path.join(fdir, f"f_{f:04d}.png")
            bpy.ops.render.render(write_still=True)
            if args["post_bloom"]:
                post_bloom(scene.render.filepath)
            print(f"frame {f} done {time.time()-tr:.1f}s")
        return
    scene.render.filepath = os.path.join(fdir, "f_")
    bpy.ops.render.render(animation=True)
    print(f"rendered {nfr} frames in {time.time()-tr:.1f}s")
    if args["post_bloom"]:
        for f in range(1, nfr + 1):
            post_bloom(os.path.join(fdir, f"f_{f:04d}.png"))
    ff = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    mp4 = os.path.join(out, f"m2nwu_{args['shot']}_{args['camera']}.mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(fdir, "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", mp4], check=False)
    print("mp4:", mp4)


if __name__ == "__main__":
    main()
