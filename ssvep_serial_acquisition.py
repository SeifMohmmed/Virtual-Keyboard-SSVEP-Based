"""
SSVEP Real-Time Serial Acquisition Layer
=========================================
Connects STM32 NUCLEO-F303K8 → BioAmp EXG serial stream to the existing
RealTimeSSVEPPipeline from ssvep_pipeline.py.

Data format expected from STM32:
    Raw binary: 2-byte little-endian ADC samples streamed continuously.
    e.g. bytes \xe1\x07 = 0x07E1 = 2017 (12-bit ADC value)

Dependencies:
    pip install pyserial numpy

Run:
    python ssvep_serial_acquisition.py --port COM12 --baud 57600

    # Linux / macOS:
    python ssvep_serial_acquisition.py --port /dev/ttyACM0 --baud 57600

    # Dry-run with simulated serial (no hardware needed):
    python ssvep_serial_acquisition.py --simulate

FIXES (2025-03-27)
------------------
FIX-1: ConfidenceSmoother.__init__ — removed bogus `vote_window` parameter
       that was causing a TypeError on construction, silently disabling the
       smoother entirely. Parameters now match the EMA design: alpha,
       confirm_threshold, cooldown_sec, min_snr_db.

FIX-2: ConfidenceSmoother.push() — corrected call signature.
       Previously _main_loop called push(prediction=..., snr_db=..., quality_ok=...)
       but the smoother expected push(snr_per_freq: dict, quality_ok: bool).
       Now accepts the full `info` dict from the pipeline plus the raw
       prediction, which gives it access to snr_fundamental (per-frequency
       SNR list) for the softmax — identical to what run_bci.py's smoother
       uses. Scalar snr_db path is gone.

FIX-3: ConfidenceSmoother.push() — scores now reset to zero after every
       confirmed selection. Without this, stale evidence from the previous
       selection immediately re-triggers on the next window.

FIX-4: SSVEPAcquisitionSession.__init__ — smoother now constructed with
       correct parameter names (alpha, confirm_threshold, cooldown_sec,
       min_snr_db) and receives stim_freqs so the EMA dict is keyed
       correctly.

FIX-5: _main_loop — smoother.push() call updated to new signature.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lazy pyserial import so the module loads cleanly in --simulate mode
# ---------------------------------------------------------------------------
try:
    import serial
    import serial.serialutil
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Import the existing pipeline (must be in the same directory or on PYTHONPATH)
# ---------------------------------------------------------------------------
try:
    from ssvep_pipeline import (
        RealTimeSSVEPPipeline,
        SSVEPConfig,
        parse_stm32_csv,
    )
except ImportError as exc:
    sys.exit(
        f"[ERROR] Cannot import ssvep_pipeline.py: {exc}\n"
        "Make sure ssvep_pipeline.py is in the same directory."
    )


# =============================================================================
# LOGGING
# =============================================================================

def _build_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# =============================================================================
# BINARY PARSER  — STM32 raw 2-byte little-endian ADC samples
# =============================================================================

def parse_stm32_binary(data: bytes) -> List[int]:
    """
    Parse a raw binary buffer from STM32.
    Each sample is 2 bytes, little-endian (e.g. \xe1\x07 = 0x07E1 = 2017).
    Returns a list of integer ADC values.
    Any trailing odd byte is discarded (incomplete sample).
    """
    samples = []
    for i in range(0, len(data) - 1, 2):
        value = int.from_bytes(data[i:i + 2], byteorder='little')
        # Clamp to valid 12-bit ADC range
        value = max(0, min(value, 4095))
        samples.append(value)
    return samples


# =============================================================================
# DECISION SMOOTHER
# =============================================================================

class ConfidenceSmoother:
    """
    Exponential moving average over softmax-normalised per-frequency SNR.

    Parameters
    ----------
    stim_freqs        : list of stimulus frequencies (Hz) — must match
                        SSVEPConfig.STIM_FREQS so the EMA dict aligns with
                        the snr_fundamental list returned by the pipeline.
    alpha             : EMA weight for the current window (0–1).
                        0.35 → roughly the last 3 windows dominate (~1.5 s
                        at 0.5 s step), fast enough to confirm in ~2 s.
    confirm_threshold : minimum EMA score for the leading frequency before
                        a confirmation fires (0–1).  0.55 is a good balance
                        between speed and false-positive rate.
    cooldown_sec      : minimum gap between consecutive confirmations.
    min_snr_db        : windows whose best SNR is below this are treated as
                        noise — scores decay but do not update.
    """

    def __init__(
        self,
        stim_freqs: List[float],
        alpha: float = 0.35,
        confirm_threshold: float = 0.55,
        cooldown_sec: float = 1.0,
        min_snr_db: float = 3.5,
    ):
        # FIX-1: stim_freqs is now a required first argument (was missing).
        # Removed bogus `vote_window` kwarg that caused TypeError on construction.
        self.stim_freqs        = list(stim_freqs)
        self.alpha             = alpha
        self.confirm_threshold = confirm_threshold
        self.cooldown_sec      = cooldown_sec
        self.min_snr_db        = min_snr_db
        self._scores           = {f: 0.0 for f in stim_freqs}
        self._last_confirm     = 0.0

    # FIX-2: push() now accepts (info: dict, raw_prediction) matching the
    # pipeline's output.  Previously took (prediction, snr_db, quality_ok)
    # which gave the smoother only a scalar SNR — useless for softmax over
    # per-frequency scores.
    def push(self, info: dict, raw_prediction: Optional[float]) -> Optional[float]:
        """
        Update EMA scores with the latest window result.

        Parameters
        ----------
        info           : dict returned by SSVEPFeatureExtractor.extract().
                         Must contain 'snr_fundamental', 'best_snr_db',
                         and 'quality_ok'.
        raw_prediction : frequency predicted by the pipeline classifier,
                         or None if the window was rejected.

        Returns
        -------
        Confirmed frequency (float) if the EMA crosses the threshold, else None.
        """
        quality_ok = info.get("quality_ok", False)
        best_snr   = info.get("best_snr_db", -99.0)

        # Bad quality or SNR below floor → decay, don't update
        if not quality_ok or best_snr < self.min_snr_db:
            for f in self.stim_freqs:
                self._scores[f] *= (1.0 - self.alpha)
            return None

        # Softmax over per-frequency SNR → probability distribution
        snr_vals = np.array([
            max(s, 0.0)
            for s in info.get("snr_fundamental", [0.0] * len(self.stim_freqs))
        ])
        exp_snr = np.exp(snr_vals - snr_vals.max())   # numerically stable
        probs   = exp_snr / (exp_snr.sum() + 1e-9)

        # EMA update
        for i, f in enumerate(self.stim_freqs):
            self._scores[f] = (
                (1.0 - self.alpha) * self._scores[f] + self.alpha * float(probs[i])
            )

        best_freq  = max(self._scores, key=self._scores.get)
        best_score = self._scores[best_freq]

        now = time.monotonic()
        if best_score >= self.confirm_threshold and \
                (now - self._last_confirm) >= self.cooldown_sec:
            self._last_confirm = now
            # FIX-3: reset scores after confirm so stale evidence doesn't
            # immediately re-trigger on the very next window.
            for f in self.stim_freqs:
                self._scores[f] = 0.0
            return best_freq

        return None

    def reset(self) -> None:
        self._scores       = {f: 0.0 for f in self.stim_freqs}
        self._last_confirm = 0.0


# =============================================================================
# DATA LOGGER  (CSV writer for raw + processed data)
# =============================================================================

class DataLogger:
    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_path  = output_dir / f"raw_{tag}.csv"
        proc_path = output_dir / f"proc_{tag}.csv"

        self._raw_fh  = open(raw_path,  "w", newline="", encoding="utf-8")
        self._proc_fh = open(proc_path, "w", newline="", encoding="utf-8")

        self._raw_writer  = csv.writer(self._raw_fh)
        self._proc_writer = csv.writer(self._proc_fh)

        self._raw_writer.writerow(["sample_idx", "adc_value"])
        self._proc_writer.writerow([
            "wall_time_s", "window_idx",
            "best_candidate_hz", "best_snr_db", "entropy",
            "quality_ok", "raw_prediction_hz", "confirmed_prediction_hz",
        ])

        self._lock = threading.Lock()
        self._raw_path  = raw_path
        self._proc_path = proc_path

    def log_raw(self, sample_idx: int, adc_value: int) -> None:
        with self._lock:
            self._raw_writer.writerow([sample_idx, adc_value])

    def log_result(
        self,
        window_idx: int,
        info: dict,
        raw_prediction: Optional[float],
        confirmed: Optional[float],
    ) -> None:
        with self._lock:
            self._proc_writer.writerow([
                f"{time.time():.4f}",
                window_idx,
                f"{info['best_candidate']:.1f}",
                f"{info['best_snr_db']:.3f}",
                f"{info['spectral_entropy']:.4f}",
                int(info["quality_ok"]),
                raw_prediction if raw_prediction is not None else "",
                confirmed if confirmed is not None else "",
            ])

    def flush(self) -> None:
        with self._lock:
            self._raw_fh.flush()
            self._proc_fh.flush()

    def close(self) -> None:
        with self._lock:
            self._raw_fh.close()
            self._proc_fh.close()

    @property
    def paths(self) -> Tuple[Path, Path]:
        return self._raw_path, self._proc_path


# =============================================================================
# BINARY SERIAL READER  (producer thread)
# =============================================================================

class SerialReader:
    """
    Reads raw binary data from STM32.
    Each ADC sample is 2 bytes little-endian.
    Accumulates bytes and extracts complete 2-byte samples into the queue.
    """

    def __init__(
        self,
        port: str,
        baud: int,
        shared_queue: Deque[Tuple[float, int]],
        logger: logging.Logger,
        data_logger: Optional[DataLogger] = None,
        stop_event: Optional[threading.Event] = None,
        reconnect_delay_sec: float = 2.0,
    ):
        self.port               = port
        self.baud               = baud
        self._queue             = shared_queue
        self._log               = logger
        self._data_logger       = data_logger
        self._stop              = stop_event or threading.Event()
        self._reconnect_delay   = reconnect_delay_sec

        self.lines_received: int = 0
        self.lines_parsed:   int = 0
        self.lines_dropped:  int = 0

        self._thread = threading.Thread(
            target=self._run, name="SerialReader", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 3.0) -> None:
        self._thread.join(timeout=timeout)

    def _open_port(self) -> "serial.Serial":
        while not self._stop.is_set():
            try:
                port = serial.Serial(
                    port=self.port,
                    baudrate=self.baud,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=1.0,
                    write_timeout=1.0,
                    xonxoff=False,
                    rtscts=False,
                )
                port.reset_input_buffer()
                self._log.info(
                    "Serial port opened: %s @ %d baud", self.port, self.baud
                )
                return port
            except serial.SerialException as exc:
                self._log.warning(
                    "Cannot open %s: %s — retrying in %.1f s",
                    self.port, exc, self._reconnect_delay,
                )
                time.sleep(self._reconnect_delay)
        raise RuntimeError("Stop requested before port could be opened.")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                port = self._open_port()
                self._read_loop(port)
            except RuntimeError:
                break
            except Exception as exc:
                self._log.error("Unexpected reader error: %s", exc, exc_info=True)
                time.sleep(self._reconnect_delay)

        self._log.info("SerialReader thread exiting.")

    def _read_loop(self, port: "serial.Serial") -> None:
        """
        Read raw binary data in chunks.
        Each sample = 2 bytes little-endian.
        We use a leftover buffer to handle partial reads.
        """
        self._log.info("Starting read loop on %s (binary mode)", port.name)
        leftover = b""
        sample_idx = 0

        try:
            while not self._stop.is_set():
                try:
                    # Read available bytes (up to 256 at a time)
                    raw_bytes = port.read(256)
                except serial.SerialException as exc:
                    self._log.warning("Serial read error: %s — reconnecting.", exc)
                    break

                if not raw_bytes:
                    continue

                self.lines_received += len(raw_bytes)

                # Combine leftover from previous read with new data
                data = leftover + raw_bytes
                leftover = b""

                # Extract complete 2-byte samples
                i = 0
                while i + 1 < len(data):
                    value = int.from_bytes(data[i:i + 2], byteorder='little')
                    value = max(0, min(value, 4095))  # clamp to 12-bit

                    self._queue.append((sample_idx / 500.0, value))
                    self.lines_parsed += 1

                    if self._data_logger is not None:
                        self._data_logger.log_raw(sample_idx, value)

                    sample_idx += 1
                    i += 2

                # Save any trailing odd byte for the next read
                if i < len(data):
                    leftover = data[i:]

        finally:
            port.close()
            self._log.info("Serial port closed.")


# =============================================================================
# SIMULATED SERIAL READER
# =============================================================================

class SimulatedReader:
    """
    Drops synthetic SSVEP data into the shared_queue at the correct real-time rate.
    """

    def __init__(
        self,
        shared_queue: Deque[Tuple[float, int]],
        logger: logging.Logger,
        fs: int = 500,
        target_freq: float = 10.0,
        stop_event: Optional[threading.Event] = None,
    ):
        self._queue  = shared_queue
        self._log    = logger
        self._fs     = fs
        self._target = target_freq
        self._stop   = stop_event or threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="SimulatedReader", daemon=True
        )
        self.lines_received = 0
        self.lines_parsed   = 0
        self.lines_dropped  = 0

    def start(self) -> None:
        self._log.info(
            "SimulatedReader: generating %g Hz SSVEP at %d Hz sample rate.",
            self._target, self._fs,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 3.0) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        rng        = np.random.default_rng(0)
        fs         = self._fs
        f          = self._target
        vref_mv    = 3.3 * 1000.0
        adc_max    = (1 << 12) - 1
        interval   = 1.0 / fs
        t          = 0.0
        t_start    = time.monotonic()
        sample_idx = 0

        while not self._stop.is_set():
            target_wall = t_start + sample_idx * interval
            now = time.monotonic()
            sleep_dur = target_wall - now
            if sleep_dur > 0:
                time.sleep(sleep_dur)

            signal_uv = (
                50.0 * np.sin(2 * np.pi * f * t)
                + 20.0 * np.sin(2 * np.pi * 2 * f * t)
                + 5.0  * np.sin(2 * np.pi * 3 * f * t)
                + rng.normal(0, 30)
                + 200.0 * np.sin(2 * np.pi * 50.0 * t)
                + 100.0 + 50.0 * np.sin(2 * np.pi * 0.05 * t)
            )
            mv  = signal_uv / 1000.0
            adc = int(np.clip(round(mv / vref_mv * adc_max), 0, adc_max))

            self._queue.append((t, adc))
            self.lines_received += 1
            self.lines_parsed   += 1

            t          += interval
            sample_idx += 1

        self._log.info("SimulatedReader thread exiting.")


# =============================================================================
# MAIN ACQUISITION LOOP
# =============================================================================

class SSVEPAcquisitionSession:
    def __init__(
        self,
        port: Optional[str],
        baud: int = 57600,
        cfg: Optional[SSVEPConfig] = None,
        chunk_size: int = 100,          # raised from 50 — helps drain queue
        queue_maxlen: int = 5000,
        simulate: bool = False,
        log_dir: Optional[Path] = None,
        # FIX-4: removed `vote_window` (unused), added `alpha` and
        # `confirm_threshold` so callers can tune the EMA directly.
        alpha: float = 0.35,
        confirm_threshold: float = 0.55,
        cooldown_sec: float = 1.0,
        min_snr_db: float = 3.5,        # raised from 2.0
        verbose: bool = False,
    ):
        if cfg is None:
            cfg = SSVEPConfig(fs=500, stim_freqs=[8.0, 10.0, 12.0, 15.0])

        self.cfg        = cfg
        self.chunk_size = chunk_size
        self.verbose    = verbose

        log_file = (log_dir / "session.log") if log_dir else None
        self._log = _build_logger("SSVEPAcq", log_file=log_file)

        self._data_logger = DataLogger(log_dir) if log_dir else None

        self._queue: Deque[Tuple[float, int]] = deque(maxlen=queue_maxlen)
        self._stop_event = threading.Event()

        if simulate:
            self._reader = SimulatedReader(
                shared_queue=self._queue,
                logger=self._log,
                fs=cfg.FS,
                stop_event=self._stop_event,
            )
        else:
            if not SERIAL_AVAILABLE:
                sys.exit("[ERROR] pyserial is not installed. Run: pip install pyserial")
            if port is None:
                sys.exit("[ERROR] --port is required when not using --simulate")
            self._reader = SerialReader(
                port=port,
                baud=baud,
                shared_queue=self._queue,
                logger=self._log,
                data_logger=self._data_logger,
                stop_event=self._stop_event,
            )

        self._pipeline = RealTimeSSVEPPipeline(cfg)

        # FIX-4: correct constructor — stim_freqs first, then EMA params.
        # Previously passed `vote_window=` which doesn't exist → TypeError.
        self._smoother = ConfidenceSmoother(
            stim_freqs=cfg.STIM_FREQS,
            alpha=alpha,
            confirm_threshold=confirm_threshold,
            cooldown_sec=cooldown_sec,
            min_snr_db=min_snr_db,
        )

        self._window_idx     = 0
        self._session_start  = 0.0

    def run(self) -> None:
        self._session_start = time.monotonic()
        self._log.info("=" * 60)
        self._log.info("SSVEP Acquisition Session starting")
        self._log.info("  Stimuli     : %s Hz", self.cfg.STIM_FREQS)
        self._log.info("  Window      : %.1f s (%d samples)", self.cfg.WINDOW_SEC, self.cfg.WINDOW_SAMPLES)
        self._log.info("  Step        : %.1f s", self.cfg.WINDOW_SEC * (1 - self.cfg.OVERLAP))
        self._log.info("  Chunk size  : %d samples", self.chunk_size)
        self._log.info("  Smoother α  : %.2f  confirm ≥ %.2f  cooldown %.1f s",
                       self._smoother.alpha,
                       self._smoother.confirm_threshold,
                       self._smoother.cooldown_sec)
        self._log.info("  MIN_SNR_DB  : %.1f dB", self._smoother.min_snr_db)
        if self._data_logger:
            raw_p, proc_p = self._data_logger.paths
            self._log.info("  Raw log     : %s", raw_p)
            self._log.info("  Proc log    : %s", proc_p)
        self._log.info("=" * 60)
        self._log.info("Press Ctrl-C to stop.\n")

        self._reader.start()

        try:
            self._main_loop()
        except KeyboardInterrupt:
            self._log.info("\nCtrl-C received — shutting down.")
        finally:
            self._shutdown()

    def _main_loop(self) -> None:
        chunk_accumulator: List[int] = []
        last_stats_time = time.monotonic()

        while not self._stop_event.is_set():

            while self._queue and len(chunk_accumulator) < self.chunk_size:
                _ts, adc = self._queue.popleft()
                chunk_accumulator.append(adc)

            if len(chunk_accumulator) < self.chunk_size:
                time.sleep(0.0005)
                continue

            chunk_np = np.array(chunk_accumulator, dtype=np.int32)
            chunk_accumulator.clear()

            result = self._pipeline.push(chunk_np)

            if result is None:
                continue

            fv, info, raw_prediction = result
            self._window_idx += 1

            # FIX-5: pass (info, raw_prediction) — smoother now reads the
            # full per-frequency snr_fundamental list for softmax, not just
            # a scalar best_snr_db.
            confirmed = self._smoother.push(info, raw_prediction)

            if self._data_logger:
                self._data_logger.log_result(
                    self._window_idx, info, raw_prediction, confirmed
                )

            self._print_result(info, raw_prediction, confirmed)

            now = time.monotonic()
            if now - last_stats_time >= 10.0:
                self._print_stats()
                last_stats_time = now

            if self._data_logger and self._window_idx % 20 == 0:
                self._data_logger.flush()

    def _print_result(
        self,
        info: dict,
        raw_prediction: Optional[float],
        confirmed: Optional[float],
    ) -> None:
        snr     = info["best_snr_db"]
        entropy = info["spectral_entropy"]
        cand    = info["best_candidate"]
        quality = "OK    " if info["quality_ok"] else "REJECT"

        if confirmed is not None:
            elapsed = time.monotonic() - self._session_start
            print(
                f"\n{'='*50}\n"
                f"  *** CONFIRMED: {confirmed:.1f} Hz ***\n"
                f"  SNR: {snr:+.2f} dB  |  Entropy: {entropy:.3f}"
                f"  |  t = {elapsed:.1f} s\n"
                f"{'='*50}\n",
                flush=True,
            )
        elif self.verbose or raw_prediction is not None:
            pred_str = f"{raw_prediction:.1f} Hz" if raw_prediction is not None else "—    "
            print(
                f"[W{self._window_idx:04d}] best={cand:.1f}Hz "
                f"SNR={snr:+6.2f}dB  H={entropy:.3f}  {quality}  "
                f"raw={pred_str}",
                flush=True,
            )

    def _print_stats(self) -> None:
        elapsed = time.monotonic() - self._session_start
        q_len   = len(self._queue)
        drop_r  = self._reader.lines_dropped
        parse_r = self._reader.lines_parsed
        parse_pct = (parse_r / max(self._reader.lines_received, 1)) * 100
        self._log.info(
            "Stats | t=%.0fs  windows=%d  queue_depth=%d  "
            "parsed=%.1f%%  dropped_lines=%d",
            elapsed, self._window_idx, q_len, parse_pct, drop_r,
        )

    def _shutdown(self) -> None:
        self._stop_event.set()
        self._reader.stop()
        self._reader.join(timeout=3.0)
        if self._data_logger:
            self._data_logger.flush()
            self._data_logger.close()
            raw_p, proc_p = self._data_logger.paths
            self._log.info("Logs saved:\n  Raw : %s\n  Proc: %s", raw_p, proc_p)
        elapsed = time.monotonic() - self._session_start
        self._log.info(
            "Session ended. Duration: %.1f s  |  Windows processed: %d",
            elapsed, self._window_idx,
        )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SSVEP real-time serial acquisition — STM32 + BioAmp EXG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    serial_grp = p.add_argument_group("serial")
    serial_grp.add_argument("--port", "-p", type=str, default=None)
    serial_grp.add_argument("--baud", "-b", type=int, default=57600)
    serial_grp.add_argument("--simulate", action="store_true")

    pipe_grp = p.add_argument_group("pipeline")
    pipe_grp.add_argument("--fs", type=int, default=500)
    pipe_grp.add_argument("--freqs", type=float, nargs="+", default=[8.0, 10.0, 12.0, 15.0])
    pipe_grp.add_argument("--chunk", type=int, default=100)
    pipe_grp.add_argument("--window", type=float, default=2.0)
    pipe_grp.add_argument("--overlap", type=float, default=0.5)

    smooth_grp = p.add_argument_group("decision smoother")
    smooth_grp.add_argument("--alpha", type=float, default=0.35,
                            help="EMA weight per window (0–1)")
    smooth_grp.add_argument("--confirm", type=float, default=0.55,
                            help="EMA score threshold to confirm a selection")
    smooth_grp.add_argument("--cooldown", type=float, default=1.0)
    smooth_grp.add_argument("--min-snr", type=float, default=3.5)

    out_grp = p.add_argument_group("output")
    out_grp.add_argument("--log-dir", type=Path, default=None)
    out_grp.add_argument("--verbose", "-v", action="store_true")

    return p


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    cfg = SSVEPConfig(
        fs=args.fs,
        stim_freqs=args.freqs,
        WINDOW_SEC=args.window,
        OVERLAP=args.overlap,
    )

    session = SSVEPAcquisitionSession(
        port=args.port,
        baud=args.baud,
        cfg=cfg,
        chunk_size=args.chunk,
        simulate=args.simulate,
        log_dir=args.log_dir,
        alpha=args.alpha,
        confirm_threshold=args.confirm,
        cooldown_sec=args.cooldown,
        min_snr_db=args.min_snr,
        verbose=args.verbose,
    )

    session.run()


if __name__ == "__main__":
    main()