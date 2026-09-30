"""BetterDeclipper: offline, high-accuracy audio declipping.

Python API (torch is imported on first use):
    import soundfile as sf
    from betterdeclipper import declip
    y, sr = sf.read("in.wav", always_2d=True)
    x, info = declip(y, sr, preset="normal", mode="auto", progress=lambda step, n, seconds: None)
info["analysis"] holds the automatic analysis; betterdeclipper.cli.analysis_lines(info, channels) formats it.
"""
__version__ = "0.2.0"
__all__ = ["declip", "PRESETS", "__version__"]


def __getattr__(name):  # lazy, so that `import betterdeclipper` stays light
    if name in ("declip", "PRESETS"):
        from . import engine
        return getattr(engine, name)
    raise AttributeError(f"module 'betterdeclipper' has no attribute {name!r}")
