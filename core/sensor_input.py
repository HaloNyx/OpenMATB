"""
Sensor Input Module — Physiological Data for AGSM Assessment
=============================================================

This module provides a clean hardware-abstraction interface for reading
physiological signals from a participant during Anti-G Straining Maneuver
(AGSM) trials.  It ships with a *simulated* implementation that returns
realistic placeholder values so that the rest of the system can be
developed and tested without physical hardware.

HARDWARE INTEGRATION
--------------------
When the spirometer and EMG equipment arrive, create a subclass of
:class:`SensorInput` that wraps your device driver, then register it
once at experiment startup::

    from core.sensor_input import set_sensor
    from mylab.hardware import SpirometerEMGDevice
    set_sensor(SpirometerEMGDevice(port="/dev/ttyUSB0"))

Everything else in the system calls :func:`get_sensor()` and is
unaffected by the swap.

Expected hardware signals
~~~~~~~~~~~~~~~~~~~~~~~~~
* **Spirometer** (facemask) — provides:
  - Respiratory depth  (normalised tidal/vital-capacity fraction, 0–1)
  - Respiratory force  (normalised mouth-pressure, 0–1; Valsalva during AGSM)
  - Respiratory frequency (breaths per minute, derived from flow signal)

* **Surface EMG** — provides:
  - MVIC %  (% of Maximum Voluntary Isometric Contraction established
             during pre-experiment calibration, 0–100)

Expert AGSM target ranges (placeholder — update with study data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+--------------------+-----------+-----------+
| Measure            | Target    | Soft edge |
+====================+===========+===========+
| MVIC %             | 70 – 100  | 50 – 100  |
+--------------------+-----------+-----------+
| Respiratory force  | 30 – 65 % | 15 – 80 % |
+--------------------+-----------+-----------+
| Respiratory freq.  | 4 – 8 bpm | 2 – 14    |
+--------------------+-----------+-----------+
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Optional


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class SensorInput(ABC):
    """
    Abstract base class defining the physiological sensor interface.

    Subclass this and implement the four abstract methods to connect real
    hardware.  The :meth:`compute_agsm_quality` scoring logic is shared by
    all implementations and should not normally need to be overridden.
    """

    @abstractmethod
    def get_respiratory_depth(self) -> float:
        """
        Normalised inhalation depth (0.0 = no effort, 1.0 = full capacity).

        Typically the instantaneous tidal volume expressed as a fraction of
        the participant's forced vital capacity (FVC) measured at calibration.
        Source: spirometer flow integration.
        """

    @abstractmethod
    def get_respiratory_force(self) -> float:
        """
        Normalised respiratory pressure / force (0.0 = none, 1.0 = maximum).

        During AGSM the participant performs a Valsalva-like manoeuvre, so
        this value rises significantly above quiet-breathing baseline.
        Source: spirometer pressure transducer.
        """

    @abstractmethod
    def get_respiratory_frequency(self) -> float:
        """
        Breathing frequency in breaths per minute (bpm).

        Expert AGSM uses a slow, controlled 'L-1' breathing pattern (~5 bpm).
        Source: zero-crossing detection on the spirometer flow signal.
        """

    @abstractmethod
    def get_mvic_percent(self) -> float:
        """
        EMG-measured isometric contraction as % of MVIC (0.0 – 100.0).

        Baseline MVIC is established during a pre-experiment calibration
        squeeze.  During AGSM the participant strains large muscle groups
        (abdomen, legs) to push blood toward the brain.
        Source: surface EMG electrode(s).
        """

    def update(self) -> None:
        """
        Called once per sensor-polling cycle (default: every 50 ms).

        Override if your hardware requires explicit polling rather than
        callback/streaming.  The default implementation is a no-op (suitable
        for callback-driven or always-fresh simulation data).
        """

    # ------------------------------------------------------------------
    # Shared scoring — do not normally need to override
    # ------------------------------------------------------------------

    def compute_agsm_quality(self) -> float:
        """
        Return an AGSM quality score in [0.0, 1.0].

        Combines the three physiological dimensions into a single metric
        that reflects how closely the participant's values match those of
        expert pilots performing a correct AGSM.

        +----------+---------+----------------------------------------------+
        | Score    | Meaning | Typical greyout effect                       |
        +==========+=========+==============================================+
        | ≥ 0.70   | Good    | Greyout recovers at ``recovery_rate``/s      |
        +----------+---------+----------------------------------------------+
        | 0.30–0.70| Partial | Greyout slows; may stabilise                 |
        +----------+---------+----------------------------------------------+
        | < 0.30   | Poor    | Greyout worsens at full ``onset_rate``/s     |
        +----------+---------+----------------------------------------------+
        """
        mvic       = self.get_mvic_percent()
        resp_force = self.get_respiratory_force() * 100.0   # scale → 0–100
        resp_freq  = self.get_respiratory_frequency()

        mvic_score  = _score_range(mvic,       70, 100, 50, 100)
        force_score = _score_range(resp_force, 30,  65, 15,  80)
        freq_score  = _score_range(resp_freq,   4,   8,  2,  14)

        # MVIC carries the most weight for G-force protection
        quality = 0.50 * mvic_score + 0.30 * force_score + 0.20 * freq_score
        return float(max(0.0, min(1.0, quality)))


def _score_range(
    value: float,
    target_lo: float, target_hi: float,
    soft_lo:   float, soft_hi:   float,
) -> float:
    """
    Return a score in [0, 1] based on proximity to a target range.

    Full score (1.0) inside [target_lo, target_hi]; tapers linearly to 0.0
    at the soft boundaries; 0.0 outside them.
    """
    if target_lo <= value <= target_hi:
        return 1.0
    elif soft_lo <= value < target_lo:
        return (value - soft_lo) / max(1e-9, target_lo - soft_lo)
    elif target_hi < value <= soft_hi:
        return (soft_hi - value) / max(1e-9, soft_hi - target_hi)
    return 0.0


# ---------------------------------------------------------------------------
# Simulated (placeholder) implementation
# ---------------------------------------------------------------------------

class SimulatedSensorInput(SensorInput):
    """
    **PLACEHOLDER** — simulated physiological data for development & testing.

    Reproduces a realistic AGSM response timeline referenced from the cue:

    +-------------+-----------------------------------------------------------+
    | Phase       | Description                                               |
    +=============+===========================================================+
    | t < 0       | Cue not yet delivered — resting baseline only             |
    +-------------+-----------------------------------------------------------+
    | 0 – 1.8 s   | Reaction window — values still resting                   |
    +-------------+-----------------------------------------------------------+
    | 1.8 – 4.8 s | Smooth ramp-up as participant begins to strain            |
    +-------------+-----------------------------------------------------------+
    | > 4.8 s     | Sustained AGSM with realistic physiological variability   |
    +-------------+-----------------------------------------------------------+

    Replace this class (or call :func:`set_sensor`) with a hardware
    implementation when equipment is available.

    Simulated sensor values
    ~~~~~~~~~~~~~~~~~~~~~~~
    Resting baseline:
      - MVIC: 6 %,  Resp. force: 0.07,  Resp. freq: 14 bpm

    Full AGSM target:
      - MVIC: 82 %, Resp. force: 0.45,  Resp. freq:  5 bpm
    """

    # -- Expert AGSM target values (update once you have real pilot data) --
    _EXPERT_MVIC       = 82.0    # % MVIC
    _EXPERT_RESP_FORCE = 0.45    # normalised pressure
    _EXPERT_RESP_FREQ  = 5.0     # bpm

    # -- Resting baseline values -------------------------------------------
    _REST_MVIC       = 6.0
    _REST_RESP_FORCE = 0.07
    _REST_RESP_FREQ  = 14.0

    # -- Timing -----------------------------------------------------------
    _REACTION_TIME_S  = 1.8   # s before strain begins after cue
    _RAMP_DURATION_S  = 3.0   # s for full engagement to develop

    def __init__(self) -> None:
        self._cue_time: Optional[float] = None

    def notify_agsm_cue(self) -> None:
        """Signal that the AGSM audio cue has just been played."""
        self._cue_time = time.monotonic()

    def update(self) -> None:
        pass  # Simulation is stateless (computed on-demand each call)

    # -- Internal helpers -------------------------------------------------

    def _elapsed(self) -> Optional[float]:
        if self._cue_time is None:
            return None
        return time.monotonic() - self._cue_time

    def _engagement(self) -> float:
        """
        Engagement factor: 0.0 (resting) → ~1.0 (full AGSM).
        Uses a smooth-step curve after the reaction-time window.
        """
        t = self._elapsed()
        if t is None or t < self._REACTION_TIME_S:
            return 0.0

        phase = min(1.0, (t - self._REACTION_TIME_S) / self._RAMP_DURATION_S)

        # Smooth-step: 3t² − 2t³  (C1-continuous, zero derivative at ends)
        f = phase * phase * (3.0 - 2.0 * phase)

        # At full engagement add mild physiological variability
        if phase >= 1.0:
            f = max(0.0, min(1.0, f + 0.04 * math.sin(time.monotonic() * 1.7)))

        return f

    def _raw_breath_freq(self) -> float:
        """Breathing frequency without noise (avoids recursion in _engagement)."""
        f = self._engagement()
        return self._REST_RESP_FREQ + f * (self._EXPERT_RESP_FREQ - self._REST_RESP_FREQ)

    # -- SensorInput interface --------------------------------------------

    def get_mvic_percent(self) -> float:
        f = self._engagement()
        base  = self._REST_MVIC + f * (self._EXPERT_MVIC - self._REST_MVIC)
        # High-frequency EMG noise proportional to engagement
        noise = 3.5 * math.sin(time.monotonic() * 13.0) * f
        return float(max(0.0, min(100.0, base + noise)))

    def get_respiratory_force(self) -> float:
        f   = self._engagement()
        # Modulate with breathing cycle
        freq_rad = (self._raw_breath_freq() / 60.0) * 2.0 * math.pi
        cycle    = abs(math.sin(time.monotonic() * freq_rad))
        target   = self._EXPERT_RESP_FORCE * (0.65 + 0.35 * cycle)
        base     = self._REST_RESP_FORCE + f * (target - self._REST_RESP_FORCE)
        noise    = 0.015 * math.sin(time.monotonic() * 5.3)
        return float(max(0.0, min(1.0, base + noise)))

    def get_respiratory_depth(self) -> float:
        f        = self._engagement()
        freq_rad = (self._raw_breath_freq() / 60.0) * 2.0 * math.pi
        cycle    = 0.5 + 0.5 * math.sin(time.monotonic() * freq_rad)
        target   = 0.10 + f * 0.45   # deeper breaths during AGSM
        return float(max(0.0, min(1.0, target * cycle)))

    def get_respiratory_frequency(self) -> float:
        noise = 0.4 * math.sin(time.monotonic() * 0.31)
        return float(max(1.0, self._raw_breath_freq() + noise))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_sensor: SensorInput = SimulatedSensorInput()


def get_sensor() -> SensorInput:
    """Return the active :class:`SensorInput` instance."""
    return _sensor


def set_sensor(sensor: SensorInput) -> None:
    """
    Replace the active :class:`SensorInput` implementation.

    Call this once before the scenario starts (e.g. in ``main.py``) to
    swap in a hardware-connected class::

        from core.sensor_input import set_sensor
        from mylab.hardware import MySpirometerEMG
        set_sensor(MySpirometerEMG(port="/dev/ttyUSB0"))
    """
    global _sensor
    _sensor = sensor
