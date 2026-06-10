"""
Greyout / Tunnel-Vision Overlay Plugin
=======================================

Simulates the visual symptoms of G-force-induced loss of consciousness
(G-LOC) that a pilot experiences when +Gz forces exceed their tolerance
without a correct Anti-G Straining Manoeuvre (AGSM).

Visual layers (drawn on top of everything else in the scene)
------------------------------------------------------------
1. **Vignette ring**  — peripheral vision darkens and collapses inward
2. **Grey overlay**   — colour desaturation over the remaining central field
3. **Blackout layer** — complete visual loss at high intensity

All three layers are transparent at ``intensity = 0`` (normal vision) and
progressively more opaque as intensity rises toward 1.0.

Scenario usage
--------------
::

    # Load the plugin at scenario start
    0:00:00;greyout;start

    # Play the AGSM audio cue; greyout begins 2.5 s later
    0:00:30;greyout;agsm_prompt;True

    # Optional: reset overlay between episodes
    0:01:30;greyout;reset

    # Trigger a second episode
    0:01:35;greyout;agsm_prompt;True

    # Stop plugin at scenario end
    0:02:30;greyout;stop

Optional parameter tuning (also settable via scenario file)
-----------------------------------------------------------
::

    0:00:00;greyout;onset_rate;0.30      # intensity/s without AGSM
    0:00:00;greyout;recovery_rate;0.20   # intensity/s with good AGSM

Intensity dynamics
------------------
After the onset delay the greyout evolves based on real-time AGSM quality:

* **quality ≥ 0.70** (good AGSM)  → intensity decreases at ``recovery_rate``/s
* **quality 0.30–0.70** (partial)  → slow progression, may stabilise
* **quality < 0.30** (poor/none)   → intensity increases at ``onset_rate``/s
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from core import validation
from core.constants import PATHS as P
from core.rendering import get_group, get_program, polygon_indices
from core.window import Window
from plugins.abstractplugin import AbstractPlugin

from pyglet.gl import GL_TRIANGLES
from pyglet.media import Player
from pyglet.media import load as _media_load


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class Greyout(AbstractPlugin):
    """
    Greyout / Tunnel-Vision Overlay Plugin.

    Inherits from :class:`AbstractPlugin` with ``taskplacement='invisible'``
    so it occupies no panel area, but still participates in the scenario
    update loop.  Its visual output is written directly to the main batch
    at draw-orders 25-27 (above all task widgets and modal dialogs).
    """

    # ── Rendering constants ───────────────────────────────────────────────
    _N_SEG          = 64    # Vignette ring segments (higher = smoother)
    _DRAW_GREY      = 25    # Draw order: grey desaturation layer
    _DRAW_VIGNETTE  = 26    # Draw order: tunnel-vision ring
    _DRAW_BLACKOUT  = 27    # Draw order: full blackout layer

    # ── Timing ───────────────────────────────────────────────────────────
    _ONSET_DELAY_S  = 0.5   # Seconds after audio cue starts before greyout starts. allows cue to finish playing.

    def __init__(
        self,
        label:           str = "",
        taskplacement:   str = "invisible",
        taskupdatetime:  int = 50,       # 20 Hz sensor polling
    ) -> None:
        super().__init__("", taskplacement, taskupdatetime)

        # Parameter validation (used by scenario validator)
        self.validation_dict: dict[str, Callable[..., Any]] = {
            "agsm_prompt": validation.is_boolean,
        }

        # Greyout state
        self.intensity:            float = 0.0     # 0 = clear  →  1 = blackout
        self._monitoring:          bool  = False    # True after AGSM cue is played
        self._cue_scenario_time:   Optional[float] = None
        self._player:              Optional[Player] = None

        # OpenGL vertex lists (created in start())
        self._vl_vignette:  Optional[Any] = None
        self._vl_grey:      Optional[Any] = None
        self._vl_blackout:  Optional[Any] = None

        # Scenario-adjustable parameters
        new_par: dict[str, Any] = dict(
            agsm_prompt    = False,
            onset_rate     = 0.25,   # intensity/s  — how fast greyout worsens
            recovery_rate  = 0.15,   # intensity/s  — how fast it clears
        )
        self.parameters.update(new_par)

        # Sensor input (lazy import to avoid circular deps at module level)
        from core.sensor_input import get_sensor
        self._sensor = get_sensor()

    # =========================================================================
    # Life-cycle  (override to skip the standard widget scaffolding)
    # =========================================================================

    def start(self) -> None:
        """Start the plugin and allocate overlay vertex lists."""
        self.alive   = True
        self.paused  = False
        self.visible = True   # Needed so refresh_widgets() is called every frame
        self._allocate_overlay()

    def stop(self) -> None:
        """Stop the plugin and release overlay vertex lists."""
        self.alive   = False
        self.paused  = True
        self.visible = False
        self._free_overlay()
        self._stop_audio()

    def reset(self) -> None:
        """
        Clear the greyout and stop monitoring.  Safe to call between episodes.

        Accessible as a scenario command::

            0:01:30;greyout;reset
        """
        self.intensity          = 0.0
        self._monitoring        = False
        self._cue_scenario_time = None
        self._stop_audio()

    def create_widgets(self) -> None:
        """Override: no standard task panel widgets needed."""

    # =========================================================================
    # Update loop
    # =========================================================================

    def compute_next_plugin_state(self) -> None:
        """
        Rate-limited update (every ``taskupdatetime`` ms).

        Handles:
        * triggering the AGSM audio cue when ``agsm_prompt`` is set
        * polling the sensor and adjusting ``intensity`` accordingly
        """
        if not super().compute_next_plugin_state():
            return

        # --- Trigger AGSM cue ---
        if self.parameters["agsm_prompt"]:
            self.parameters["agsm_prompt"] = False
            self._trigger_cue()

        if not self._monitoring:
            return

        # --- Onset delay (0.5 s after audio cue) ---
        elapsed = self.scenario_time - self._cue_scenario_time
        if elapsed < self._ONSET_DELAY_S:
            return

        # --- Poll sensor ---
        self._sensor.update()
        quality = self._sensor.compute_agsm_quality()

        # --- Adjust intensity ---
        dt = self.parameters["taskupdatetime"] / 1000.0

        if quality >= 0.70:
            # Good AGSM — recover
            delta = -self.parameters["recovery_rate"] * dt
        elif quality >= 0.30:
            # Partial AGSM — slow the onset proportionally
            slowdown = (quality - 0.30) / 0.40   # 0 → 1 as quality 0.3 → 0.7
            delta = self.parameters["onset_rate"] * (1.0 - slowdown) * dt
        else:
            # Poor / no AGSM — full onset rate
            delta = self.parameters["onset_rate"] * dt

        self.intensity = max(0.0, min(1.0, self.intensity + delta))

        # Performance logging (appears in the session CSV)
        self.log_performance("greyout_intensity", round(self.intensity, 4))
        self.log_performance("agsm_quality",      round(quality,        4))

    def refresh_widgets(self) -> bool:
        """
        Per-frame visual update (called every ``update()`` regardless of
        ``taskupdatetime``).  Recomputes and uploads overlay geometry.
        """
        if not self.is_visible() or self._vl_vignette is None:
            return False

        i = self.intensity
        W = Window.MainWindow.width
        H = Window.MainWindow.height

        # ── Tunnel-vision vignette ────────────────────────────────────────
        pos, col = self._vignette_geometry(i, W, H)
        self._vl_vignette.position[:] = pos
        self._vl_vignette.colors[:]   = col

        # ── Grey desaturation overlay ─────────────────────────────────────
        # Peaks at i ≈ 0.65, then fades out as blackout takes over.
        if i < 0.65:
            grey_a = int(i / 0.65 * 165)
        else:
            grey_a = int(165 * (1.0 - (i - 0.65) / 0.35))
        grey_a = max(0, min(255, grey_a))
        self._vl_grey.colors[:] = (128, 128, 128, grey_a) * 4

        # ── Blackout overlay ──────────────────────────────────────────────
        # Fades in once the tunnel is very narrow (i > 0.80).
        blk_a = int(max(0.0, (i - 0.80) / 0.20) * 255) if i > 0.80 else 0
        blk_a = max(0, min(255, blk_a))
        self._vl_blackout.colors[:] = (0, 0, 0, blk_a) * 4

        return True

    # =========================================================================
    # Audio
    # =========================================================================

    def _trigger_cue(self) -> None:
        """Play the AGSM audio prompt and begin sensor monitoring."""
        self._monitoring        = True
        self._cue_scenario_time = self.scenario_time
        self.intensity          = 0.0

        # Notify simulated sensor so its internal timeline starts
        if hasattr(self._sensor, "notify_agsm_cue"):
            self._sensor.notify_agsm_cue()

        # Play audio
        wav: Path = P["SOUNDS"] / "english" / "male" / "prompt_agsm.wav"
        if wav.exists():
            try:
                source = _media_load(str(wav), streaming=False)
                self._player = Player()
                self._player.queue(source)
                self._player.play()
            except Exception as exc:
                self.logger.log_manual_entry(
                    f"Greyout: audio playback error — {exc}"
                )
        else:
            self.logger.log_manual_entry(
                f"Greyout: prompt_agsm.wav not found at {wav}"
            )

    def _stop_audio(self) -> None:
        if self._player is not None:
            try:
                self._player.pause()
            except Exception:
                pass
            self._player = None

    # =========================================================================
    # OpenGL overlay management
    # =========================================================================

    def _allocate_overlay(self) -> None:
        """Create three persistent vertex lists for the overlay layers."""
        if Window.MainWindow is None:
            return

        program = get_program()
        batch   = Window.MainWindow.batch
        W       = Window.MainWindow.width
        H       = Window.MainWindow.height

        # -- Vignette ring: N_SEG segments × 4 triangles × 3 vertices -------
        n_verts      = self._N_SEG * 4 * 3
        pos0, col0   = self._vignette_geometry(0.0, W, H)
        self._vl_vignette = program.vertex_list(
            n_verts, GL_TRIANGLES,
            batch   = batch,
            group   = get_group(order=self._DRAW_VIGNETTE),
            position = ("f", pos0),
            colors   = ("Bn", col0),
        )

        # -- Grey desaturation layer (full screen, initially invisible) ------
        self._vl_grey = program.vertex_list_indexed(
            4, GL_TRIANGLES, polygon_indices(4),
            batch    = batch,
            group    = get_group(order=self._DRAW_GREY),
            position = ("f", (0, H, W, H, W, 0, 0, 0)),
            colors   = ("Bn", (128, 128, 128, 0) * 4),
        )

        # -- Blackout layer (full screen, initially invisible) ---------------
        self._vl_blackout = program.vertex_list_indexed(
            4, GL_TRIANGLES, polygon_indices(4),
            batch    = batch,
            group    = get_group(order=self._DRAW_BLACKOUT),
            position = ("f", (0, H, W, H, W, 0, 0, 0)),
            colors   = ("Bn", (0, 0, 0, 0) * 4),
        )

    def _free_overlay(self) -> None:
        """Delete overlay vertex lists and release GPU memory."""
        for attr in ("_vl_vignette", "_vl_grey", "_vl_blackout"):
            vl = getattr(self, attr, None)
            if vl is not None:
                vl.delete()
                setattr(self, attr, None)

    # =========================================================================
    # Vignette geometry
    # =========================================================================

    def _vignette_geometry(
        self,
        intensity: float,
        W: int,
        H: int,
    ) -> Tuple[List[float], List[int]]:
        """
        Build the vignette ring vertex data for the current ``intensity``.

        The ring is composed of two concentric annular bands:

        * **Outer band** (gradient_r → outer_r): fully opaque black.
        * **Inner band** (inner_r → gradient_r): transparent-to-opaque
          gradient giving the soft edge that characterises real tunnel vision.

        At ``intensity = 0.0``, ``inner_r`` is pushed just beyond the screen
        corners so the entire screen appears perfectly clear.

        At ``intensity = 1.0``, ``inner_r = 0`` and both bands collapse to a
        single opaque disc, producing complete blackout.

        A small stochastic radius noise appears when ``intensity > 0.75``
        to mimic the instability pilots report just before G-LOC.

        Returns
        -------
        positions : list[float]
            Flat x, y alternating list (1536 values for N_SEG = 64).
        colors : list[int]
            Flat RGBA byte list (3072 values for N_SEG = 64).
        """
        cx, cy = W / 2.0, H / 2.0

        # The clear-window radius shrinks linearly from corner-distance to 0
        corner_dist = math.sqrt(cx ** 2 + cy ** 2)
        inner_r     = corner_dist * 1.06 * max(0.0, 1.0 - intensity)

        # Near-blackout flicker: subtle radius instability (~3 % amplitude)
        if intensity > 0.75:
            noise_amp = (intensity - 0.75) / 0.25 * 0.03 * max(inner_r, 5.0)
            inner_r   = max(0.0, inner_r + noise_amp * math.sin(time.monotonic() * 23.7))

        # Gradient edge — proportional to the clear window, minimum 40 px
        grad_w     = max(40.0, inner_r * 0.38)
        gradient_r = inner_r + grad_w

        # Outer radius is always well beyond screen corners
        outer_r = corner_dist * 1.85

        N          = self._N_SEG
        positions: List[float] = []
        colors:    List[int]   = []

        OPQ  : Tuple[int, ...] = (0, 0, 0, 230)   # Opaque black at outer edge
        TRANS: Tuple[int, ...] = (0, 0, 0, 0)      # Transparent at inner edge

        for k in range(N):
            a1 = 2.0 * math.pi * k       / N
            a2 = 2.0 * math.pi * (k + 1) / N
            c1, s1 = math.cos(a1), math.sin(a1)
            c2, s2 = math.cos(a2), math.sin(a2)

            # Three concentric arc points per angle step
            ox1, oy1 = cx + outer_r    * c1, cy + outer_r    * s1
            ox2, oy2 = cx + outer_r    * c2, cy + outer_r    * s2
            gx1, gy1 = cx + gradient_r * c1, cy + gradient_r * s1
            gx2, gy2 = cx + gradient_r * c2, cy + gradient_r * s2
            ix1, iy1 = cx + inner_r    * c1, cy + inner_r    * s1
            ix2, iy2 = cx + inner_r    * c2, cy + inner_r    * s2

            # Outer band — fully opaque (2 triangles per segment)
            positions += [gx1, gy1, ox1, oy1, ox2, oy2]
            colors    += [*OPQ, *OPQ, *OPQ]
            positions += [gx1, gy1, ox2, oy2, gx2, gy2]
            colors    += [*OPQ, *OPQ, *OPQ]

            # Inner band — gradient transparent → opaque (2 triangles per segment)
            positions += [ix1, iy1, gx1, gy1, gx2, gy2]
            colors    += [*TRANS, *OPQ, *OPQ]
            positions += [ix1, iy1, gx2, gy2, ix2, iy2]
            colors    += [*TRANS, *OPQ, *TRANS]

        return positions, colors
