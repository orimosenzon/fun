"""Soundtrack synthesised from the flight recording (out/audio_data.json -> out/audio.wav).

Every engine is a sound source that moves with the bike. Its signal is built in *source
time* from the recorded spool speed and thrust:
  * turbine whine: shaft-frequency harmonics (the rotor turns at N * ~100k rpm)
  * jet roar: broadband mixing noise, level rising steeply with thrust
  * combustion rumble: low-frequency noise following fuel flow
Each shot's microphone (the camera) hears the sources at the retarded time
t_e = t - r/c, which produces the Doppler shift and propagation delay for free;
spherical spreading 1/r, air absorption of the high band, jet-noise directivity and
stereo panning from the camera's right vector. The jets hitting the ground or the lake
add a hiss/splash source at the impingement point. Shots are crossfaded at cuts.
"""
import json, pathlib, sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
SR = 48000
C = 343.0
rng = np.random.default_rng(1234)

data = json.loads((OUT / "audio_data.json").read_text())
rate = data["rate"]
dur = data["duration"]
n = int(dur * SR)
t = np.arange(n) / SR
eng = data["eng"]
T_ctrl = np.array([e["t"] for e in eng])


def up(arr):
    """resample a 120 Hz control series to audio rate (float32 keeps memory in check)"""
    return np.interp(t, T_ctrl, arr).astype(np.float32)


E = np.array([e["e"] for e in eng])          # (m, 5, 9)
H = np.array([e["hit"] for e in eng])        # (m, 5, 5)
air = np.array([e["air"] for e in eng])

def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)

def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype="low", fs=SR, output="sos"), x)

def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype="high", fs=SR, output="sos"), x)


# ---------- source signals (in source time), split in low and high bands
print("synthesising sources…", flush=True)
sources = []   # dicts: low, high, pos(3 arrays at audio rate), dir, kind
for k in range(5):
    lift = k < 4
    N = up(E[:, k, 7])
    T = up(E[:, k, 6])
    tmax = 1300.0 if lift else 1600.0
    frac = np.clip(T / tmax, 0, 1.2)
    fshaft = N * (10500 if lift else 8900) / (2 * np.pi) * (1 + 0.004 * (k - 2))   # tiny build differences
    phase = 2 * np.pi * np.cumsum(fshaft) / SR
    jitter = lp(rng.standard_normal(n), 30) * 0.6
    whine = (0.55 * np.sin(phase + jitter) + 0.28 * np.sin(2 * phase + 1.3 * jitter)
             + 0.12 * np.sin(3 * phase + 0.4) + 0.05 * np.sin(7 * phase + 2 * jitter))
    whine *= np.clip(N / 0.34, 0, 1) ** 1.5 * (0.25 + 0.35 * N)
    # roar: two noise bands, brighter with thrust
    w1 = rng.standard_normal(n)
    roar_lo = bp(w1, 60, 1100)
    roar_hi = lp(bp(rng.standard_normal(n), 1100, 5000), 2500, 1)
    lvl = frac ** 1.7
    bright = np.clip(frac, 0, 1)
    rumble = lp(rng.standard_normal(n), 120, 4) * (0.15 + frac) * (T > 1)
    # light-off: the moment thrust first appears
    lit = (T > 0.5).astype(float)
    pop = np.zeros(n)
    edges = np.where(np.diff(lit) > 0)[0]
    for e in edges:
        L = int(0.35 * SR)
        seg = lp(rng.standard_normal(L), 300) * np.exp(-np.arange(L) / (0.07 * SR)) * 2.5
        pop[e:e + L] += seg[: n - e]
    low = 1.0 * lvl * roar_lo + 0.8 * rumble + pop + 0.15 * whine * (fshaft < 1500)
    high = 0.28 * lvl * bright * roar_hi + 0.22 * whine
    pos = [up(E[:, k, i]) for i in range(3)]
    d = [up(E[:, k, 3 + i]) for i in range(3)]
    sources.append(dict(low=low.astype(np.float32), high=high.astype(np.float32), pos=pos, dir=d, kind="jet"))
    del w1, roar_lo, roar_hi, rumble, whine, phase, jitter

# jet impingement on ground/water (lift engines only): hiss or splash
for k in range(4):
    U = up(H[:, k, 3])
    wet = up(H[:, k, 4])
    s_ = np.clip((U / 260) ** 2, 0, 1.2)          # scrubbing noise ~ dynamic pressure of the jet at the surface
    hiss = lp(hp(rng.standard_normal(n), 1200), 5000) * s_ * (0.25 + 0.4 * wet)
    splash = bp(rng.standard_normal(n), 300, 3000) * s_ * wet * 0.7
    grit = bp(rng.standard_normal(n), 200, 1200) * s_ * (1 - wet) * 0.35
    pos = [up(H[:, k, i]) for i in range(3)]
    sources.append(dict(low=(splash * 0.5 + grit).astype(np.float32), high=(hiss + splash * 0.5).astype(np.float32), pos=pos, dir=None, kind="hit"))


# ---------- each shot's microphone
def render_shot(sh):
    cam = np.array(sh["cam"])
    t0 = max(0.0, sh["t0"] - 0.05)
    t1 = min(dur, sh["t1"] + 0.05)
    i0, i1 = int(t0 * SR), int(t1 * SR)
    tt = t[i0:i1]
    cx, cy, cz = (np.interp(tt, cam[:, 0], cam[:, i]) for i in (1, 2, 3))
    rx, ry, rz = (np.interp(tt, cam[:, 0], cam[:, i]) for i in (4, 5, 6))
    L = np.zeros(len(tt)); R = np.zeros(len(tt))
    for s in sources:
        px, py, pz = (a[i0:i1] for a in s["pos"])
        dx, dy, dz = px - cx, py - cy, pz - cz
        r = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        # retarded time -> Doppler
        te = tt - r / C
        lo = np.interp(te, t, s["low"])
        hi = np.interp(te, t, s["high"])
        g = 1.0 / np.maximum(r, 1.2)
        absorb = 10 ** (-0.035 * r / 20)                 # ~3.5 dB per 100 m at a few kHz
        if s["dir"] is not None:
            ex, ey, ez = (a[i0:i1] for a in s["dir"])
            # angle between the jet axis and the direction to the listener
            cosang = -(dx * ex + dy * ey + dz * ez) / r
            ang = np.arccos(np.clip(cosang, -1, 1))
            roar_dir = 0.3 + 1.0 * np.exp(-((ang - 0.6) / 0.55) ** 2)     # peak ~35 deg off the exhaust
            whine_dir = 0.55 + 0.45 * np.clip(-cosang, 0, 1)             # intake noise ahead
            sig = lo * roar_dir + hi * (roar_dir * 0.6 + whine_dir * 0.4) * absorb
        else:
            sig = lo + hi * absorb
        sig *= g
        pan = np.clip((dx * rx + dy * ry + dz * rz) / r, -1, 1)   # +1: source on the camera's right
        a = (pan + 1) * np.pi / 4
        L += sig * np.cos(a) * 1.2
        R += sig * np.sin(a) * 1.2
    if sh["onboard"]:
        # wind on a camera riding the bike
        a_air = np.interp(tt, T_ctrl, air)
        wind = lp(rng.standard_normal(len(tt)), 400) * (a_air / 40) ** 2 * 0.25
        gust = 0.7 + 0.3 * lp(rng.standard_normal(len(tt)), 3) * 8
        L += wind * gust; R += wind * gust * 0.9
    return i0, i1, L, R


print("rendering shots…", flush=True)
outL = np.zeros(n); outR = np.zeros(n); wsum = np.zeros(n)
XF = int(0.06 * SR)
rendered = [render_shot(sh) for sh in data["cams"]]
# a sound editor evens out the level between cuts: halve each shot's deviation (in dB)
lv = [np.sqrt(np.mean(L ** 2 + R ** 2)) + 1e-9 for (_, _, L, R) in rendered]
ref = np.exp(np.mean(np.log(lv)))
for sh, (i0, i1, L, R), l in zip(data["cams"], rendered, lv):
    g = (ref / l) ** 0.5
    L = L * g; R = R * g
    w = np.ones(i1 - i0)
    # crossfade windows around the cut points
    a0, a1 = int(sh["t0"] * SR) - i0, int(sh["t1"] * SR) - i0
    ramp = lambda m: 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, m))
    if sh["t0"] > 0.01:
        s0 = max(0, a0 - XF // 2)
        w[:s0] = 0
        w[s0:s0 + XF] = ramp(min(XF, len(w) - s0))
    if sh["t1"] < dur - 0.01:
        e0 = max(0, a1 - XF // 2)
        seg = w[e0:e0 + XF]
        w[e0:e0 + XF] = seg * ramp(len(seg))[::-1]
        w[e0 + XF:] = 0
    outL[i0:i1] += L * w; outR[i0:i1] += R * w; wsum[i0:i1] += w
outL /= np.maximum(wsum, 1e-3); outR /= np.maximum(wsum, 1e-3)

# ---------- ambience: valley wind + birds when the engines are quiet
engine_env = lp(np.abs(outL) + np.abs(outR), 3)
duck = np.clip(1 - engine_env / (np.percentile(engine_env, 60) + 1e-9), 0, 1)
amb = lp(rng.standard_normal(n), 500) * 0.004
birds = np.zeros(n)
for _ in range(90):
    t_b = rng.uniform(0, dur)
    i = int(t_b * SR)
    for c in range(rng.integers(2, 6)):
        L_ = int(rng.uniform(0.04, 0.12) * SR)
        f0 = rng.uniform(2800, 5200)
        sweep = f0 * (1 + rng.uniform(-0.35, 0.35) * np.linspace(0, 1, L_))
        ph = 2 * np.pi * np.cumsum(sweep) / SR
        env = np.sin(np.linspace(0, np.pi, L_)) ** 2
        j = i + c * int(rng.uniform(0.08, 0.18) * SR)
        if j + L_ < n:
            birds[j:j + L_] += np.sin(ph + 3 * np.sin(2 * np.pi * 38 * np.arange(L_) / SR)) * env * 0.004
outL += (amb + birds * duck) * 1.0
outR += (amb * 0.9 + birds * duck * 0.8)

# ---------- master: slow compressor, fades, peak normalise, soft clip
mix = np.stack([outL, outR])
env = lp(np.max(np.abs(mix), axis=0), 4) + 1e-6
thr = np.percentile(env, 90) * 0.5
gain = np.where(env > thr, (thr / env) ** (1 - 1 / 3.0), 1.0)
mix *= gain
fade = np.clip(t / 1.0, 0, 1) * np.clip((dur - t) / 1.6, 0, 1)
mix *= fade
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(mix * 1.15) / np.tanh(1.15) * 0.93
wavfile.write(OUT / "audio.wav", SR, (mix.T * 32767).astype(np.int16))
print("wrote", OUT / "audio.wav", f"{dur:.1f}s")
