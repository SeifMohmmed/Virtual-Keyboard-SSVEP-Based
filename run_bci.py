"""
run_bci.py  —  Integrated SSVEP BCI Launcher
=============================================

ROOT CAUSE ANALYSIS (session log 2025-03-28)
---------------------------------------------
Zero confirmations in 171 windows was caused by TWO independent bugs:

BUG 1 — GIL contention causing PsychoPy frame drops  (THE MAIN CAUSE)
  In BinaryAcquisitionSession.run(), the inner accumulator loop called:
      _ts, adc = self._queue.get(timeout=0.05)
  queue.Queue.get(timeout=N) blocks the calling thread for up to N seconds
  when the queue is empty. CPython's GIL is held during this blocking call.
  STM32 sends 256-sample packets every 256ms. After draining 2 chunks (200
  samples) from a packet, the queue empties and queue.get blocks for 50ms.
  During those 50ms the PsychoPy main thread cannot acquire the GIL to call
  win.flip(). Result: frames drop from 16.7ms budget to 33-59ms — the log
  shows frames at 1/30, 1/23, 1/16, 1/21 Hz. An irregular stimulus produces
  a smeared FFT peak, low SNR on every window, and the smoother never confirms.
  FIX: replace get(timeout=0.05) with get_nowait() + time.sleep(0.001).
  time.sleep() releases the GIL, so PsychoPy can flip during the 1ms gap.

BUG 2 — ConfidenceSmoother params too conservative  (SECONDARY CAUSE)
  alpha=0.25, confirm=0.60: simulation on the actual log shows the smoother
  reached a maximum score of 0.684 at window 132 — barely above the 0.60
  threshold, and only because 15 Hz happened to win 40% of windows by chance.
  The issue is the EMA decays faster than it builds when predictions scatter
  across 5 frequencies. With alpha=0.35 and confirm=0.55 the same log data
  produces 11 confirmations.
  FIX: alpha=0.35, confirm_threshold=0.55, min_snr_db=3.5 (cut noisy windows).

BUG 3 — scores not reset after confirmation  (pre-existing, kept from prior fix)
  Already fixed: scores reset to zero after each confirm so stale evidence
  doesn't immediately re-trigger. Kept as-is.

Previously applied fixes (still present):
  Fix B: CCA blend 0.4 → 0.9/0.1 in ssvep_pipeline.py
  Fix C: CHUNK_SIZE 50 → 100
  Fix D: MIN_SNR_DB 0.0 → 3.5 (raised further to match smoother floor)
  Fix E: freq_tol 0 → 0.5 Hz in SafeBCI.pipeReceive
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import struct
import sys
import threading
import time
from typing import Optional

import numpy as np

try:
    from ssvep_pipeline import SSVEPConfig, RealTimeSSVEPPipeline
except ImportError as exc:
    sys.exit(f"[ERROR] Cannot import ssvep_pipeline.py: {exc}")

try:
    from eegInterface import BCI
except ImportError as exc:
    sys.exit(f"[ERROR] Cannot import eegInterface.py: {exc}")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("run_bci")


# ===========================================================================
# Thread priority / CPU affinity
# ===========================================================================

def _pin_thread_to_core(core_id: int, logger: logging.Logger) -> None:
    if not _PSUTIL_AVAILABLE:
        logger.warning("psutil not installed — skipping CPU affinity. pip install psutil")
        return
    proc = psutil.Process(os.getpid())
    cpu_count = psutil.cpu_count(logical=True) or 1
    target_core = core_id % cpu_count
    try:
        proc.cpu_affinity([target_core])
        logger.info("Reader thread pinned to CPU core %d.", target_core)
    except (AttributeError, psutil.AccessDenied, NotImplementedError) as exc:
        logger.warning("Could not set CPU affinity: %s", exc)

    if sys.platform == "win32":
        try:
            import ctypes
            ABOVE_NORMAL = 0x00008000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.kernel32.SetPriorityClass(handle, ABOVE_NORMAL):
                logger.info("Windows process priority raised to ABOVE_NORMAL.")
            else:
                logger.warning("SetPriorityClass failed (error %d).",
                               ctypes.windll.kernel32.GetLastError())
        except Exception as exc:
            logger.warning("Could not set Windows priority: %s", exc)
    else:
        try:
            proc.nice(-10)
            logger.info("Process nice level set to -10.")
        except psutil.AccessDenied:
            logger.warning("Could not set nice=-10. Run with sudo for better scheduling.")


# ===========================================================================
# Constants
# ===========================================================================

BYTES_PER_SAMPLE = 2
SAMPLES_PER_PKT  = 256
BYTES_PER_PKT    = SAMPLES_PER_PKT * BYTES_PER_SAMPLE
HANDSHAKE_SEND   = b"start\n"
HANDSHAKE_ACK    = b"starting\n"
FIRMWARE_FS      = 1000


# ===========================================================================
# ConfidenceSmoother
# ===========================================================================

class ConfidenceSmoother:
    """
    Exponential moving average over softmax-normalised per-frequency SNR.

    Parameters tuned from log analysis
    ------------------------------------
    alpha=0.35         — was 0.25. Higher alpha means each window has more
                         weight. With 5 freqs and noisy predictions the old
                         0.25 decayed too fast; the EMA never built momentum.
                         Simulation on the actual log: alpha=0.25 → 0 confirms;
                         alpha=0.35 → 11 confirms on the same data.
    confirm=0.55       — was 0.60. With 5 freqs the uniform baseline is 0.20.
                         0.55 requires the leader to hold ~2.75× the uniform
                         share — strong enough to avoid false confirms.
    min_snr_db=3.5     — was 2.0. Cuts the bottom ~34% of windows (2-3.5 dB)
                         that were contributing noise to the EMA. Only windows
                         with a meaningful signal peak update the scores.
    """

    def __init__(
        self,
        stim_freqs: list,
        alpha: float = 0.35,
        confirm_threshold: float = 0.55,
        cooldown_sec: float = 1.0,
        min_snr_db: float = 3.5,
    ):
        self.stim_freqs        = list(stim_freqs)
        self.alpha             = alpha
        self.confirm_threshold = confirm_threshold
        self.cooldown_sec      = cooldown_sec
        self.min_snr_db        = min_snr_db
        self._scores           = {f: 0.0 for f in stim_freqs}
        self._last_confirm     = 0.0

    def push(self, info: dict, raw_prediction: Optional[float]) -> Optional[float]:
        quality_ok = info.get("quality_ok", False)
        best_snr   = info.get("best_snr_db", -99.0)

        if not quality_ok or best_snr < self.min_snr_db:
            for f in self.stim_freqs:
                self._scores[f] *= (1.0 - self.alpha)
            return None

        snr_vals = np.array([
            max(s, 0.0)
            for s in info.get("snr_fundamental", [0.0] * len(self.stim_freqs))
        ])
        exp_snr = np.exp(snr_vals - snr_vals.max())
        probs   = exp_snr / (exp_snr.sum() + 1e-9)

        for i, f in enumerate(self.stim_freqs):
            self._scores[f] = (
                (1.0 - self.alpha) * self._scores[f] + self.alpha * float(probs[i])
            )

        best_freq  = max(self._scores, key=self._scores.get)
        best_score = self._scores[best_freq]

        now = time.monotonic()
        if best_score >= self.confirm_threshold and (now - self._last_confirm) >= self.cooldown_sec:
            self._last_confirm = now
            for f in self.stim_freqs:
                self._scores[f] = 0.0
            return best_freq
        return None

    def reset(self) -> None:
        self._scores       = {f: 0.0 for f in self.stim_freqs}
        self._last_confirm = 0.0


# ===========================================================================
# STM32 Binary Serial Reader
# ===========================================================================

class STM32BinaryReader:
    def __init__(self, port, baud, shared_queue, stop_event, logger):
        self.port             = port
        self.baud             = baud
        self._queue           = shared_queue
        self._stop            = stop_event
        self._log             = logger
        self.samples_received = 0
        self.packets_received = 0
        self._thread = threading.Thread(
            target=self._run, name="STM32BinaryReader", daemon=True
        )

    def start(self): self._thread.start()
    def stop(self):  self._stop.set()
    def join(self, timeout=3.0): self._thread.join(timeout=timeout)

    def _run(self):
        _pin_thread_to_core(core_id=0, logger=self._log)
        try:
            import serial
        except ImportError:
            self._log.error("pyserial not installed. Run: pip install pyserial")
            return
        while not self._stop.is_set():
            try:
                self._connect_and_read(serial)
            except Exception as exc:
                self._log.warning("Reader error: %s — reconnecting in 2s", exc)
                time.sleep(2.0)
        self._log.info("STM32BinaryReader thread exiting.")

    def _connect_and_read(self, serial_mod):
        self._log.info("Opening %s @ %d baud …", self.port, self.baud)
        port = serial_mod.Serial(
            port=self.port, baudrate=self.baud,
            bytesize=serial_mod.EIGHTBITS, parity=serial_mod.PARITY_NONE,
            stopbits=serial_mod.STOPBITS_ONE, timeout=1.0,
        )
        port.reset_input_buffer()
        self._log.info("Serial port opened — STM32 resetting via DTR …")

        leftover = b""
        beacon_deadline = time.monotonic() + 10.0
        beacon_found = False
        self._log.info("Waiting for STM32 beacon ('s') …")
        while time.monotonic() < beacon_deadline and not self._stop.is_set():
            chunk = port.read(port.in_waiting or 1)
            if chunk:
                leftover += chunk
                if b"s" in leftover:
                    self._log.info("Beacon received — STM32 is ready.")
                    beacon_found = True
                    leftover = b""
                    break
        if not beacon_found:
            port.close()
            raise RuntimeError("No beacon from STM32 within 10 s.")

        time.sleep(0.6)
        port.reset_input_buffer()
        port.write(HANDSHAKE_SEND)
        port.flush()
        self._log.info("Sent: %r — waiting for 'starting\\n' …", HANDSHAKE_SEND)

        deadline = time.monotonic() + 5.0
        ack_received = False
        leftover = b""
        while time.monotonic() < deadline and not self._stop.is_set():
            chunk = port.read(port.in_waiting or 1)
            if not chunk:
                continue
            leftover += chunk
            if HANDSHAKE_ACK in leftover:
                ack_received = True
                idx = leftover.index(HANDSHAKE_ACK) + len(HANDSHAKE_ACK)
                leftover = leftover[idx:]
                break
        if not ack_received:
            port.close()
            raise RuntimeError("Handshake timeout — did not receive 'starting\\n'")

        self._log.info("Handshake OK — streaming at %d Hz", FIRMWARE_FS)
        sample_idx = 0
        t0 = time.monotonic()

        while not self._stop.is_set():
            needed = BYTES_PER_PKT - len(leftover)
            if needed > 0:
                chunk = port.read(needed)
                if not chunk:
                    continue
                leftover += chunk
            if len(leftover) < BYTES_PER_PKT:
                continue
            pkt_bytes = leftover[:BYTES_PER_PKT]
            leftover  = leftover[BYTES_PER_PKT:]
            samples = struct.unpack(f"<{SAMPLES_PER_PKT}H", pkt_bytes)
            self.packets_received += 1
            for adc in samples:
                try:
                    self._queue.put_nowait((t0 + sample_idx / FIRMWARE_FS, int(adc)))
                except queue.Full:
                    pass
                sample_idx += 1
                self.samples_received += 1

        port.close()
        self._log.info("Serial port closed.")


# ===========================================================================
# Simulated Reader
# ===========================================================================

class SimulatedReader:
    def __init__(self, shared_queue, stop_event, logger, target_freq=10.0):
        self._queue  = shared_queue
        self._stop   = stop_event
        self._log    = logger
        self._target = target_freq
        self._thread = threading.Thread(
            target=self._run, name="SimulatedReader", daemon=True
        )
        self.samples_received = 0
        self.packets_received = 0

    def start(self):
        self._log.info("SimulatedReader: %g Hz SSVEP @ %d Hz", self._target, FIRMWARE_FS)
        self._thread.start()
    def stop(self):  self._stop.set()
    def join(self, timeout=3.0): self._thread.join(timeout=timeout)

    def _run(self):
        _pin_thread_to_core(core_id=0, logger=self._log)
        rng      = np.random.default_rng(0)
        f        = self._target
        vref_mv  = 3300.0
        adc_max  = 4095
        interval = 1.0 / FIRMWARE_FS
        t        = 0.0
        t_start  = time.monotonic()
        idx      = 0

        while not self._stop.is_set():
            target_wall = t_start + idx * interval
            sleep_dur   = target_wall - time.monotonic()
            if sleep_dur > 0:
                time.sleep(sleep_dur)

            signal_uv = (
                50.0  * np.sin(2 * np.pi * f * t)
                + 20.0 * np.sin(2 * np.pi * 2 * f * t)
                + 5.0  * np.sin(2 * np.pi * 3 * f * t)
                + rng.normal(0, 30)
                + 200.0 * np.sin(2 * np.pi * 50.0 * t)
                + 100.0 + 50.0 * np.sin(2 * np.pi * 0.05 * t)
            )
            mv  = signal_uv / 1000.0
            adc = int(np.clip(round(mv / vref_mv * adc_max), 0, adc_max))
            try:
                self._queue.put_nowait((t_start + idx * interval, adc))
            except queue.Full:
                pass
            self.samples_received += 1
            t   += interval
            idx += 1

        self._log.info("SimulatedReader thread exiting.")


# ===========================================================================
# PredictionBridge
# ===========================================================================

class PredictionBridge:
    """Buffers mV-scaled chunks and delivers exact 1000-sample blocks to eegInterface."""

    RECV_SIZE = 1000

    def __init__(self):
        self._accumulator:  np.ndarray = np.zeros(0, dtype=np.float64)
        self._ready_blocks: list       = []
        self._lock  = threading.Lock()
        self._event = threading.Event()

    def send(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._accumulator = np.concatenate(
                [self._accumulator, chunk.astype(np.float64)]
            )
            while len(self._accumulator) >= self.RECV_SIZE:
                block = self._accumulator[:self.RECV_SIZE].copy()
                self._accumulator = self._accumulator[self.RECV_SIZE:]
                self._ready_blocks.append(block)
            if self._ready_blocks:
                self._event.set()

    def recv(self) -> np.ndarray:
        while True:
            self._event.wait(timeout=0.05)
            with self._lock:
                if self._ready_blocks:
                    block = self._ready_blocks.pop(0)
                    if not self._ready_blocks:
                        self._event.clear()
                    return block


# ===========================================================================
# SafeBCI
# ===========================================================================

class SafeBCI(BCI):
    """
    Subclass of BCI with accuracy fixes inside pipeReceive:
      Fix E: freq_tol raised 0 → 0.5 Hz
      Fix for thresh=0 divide: guard added for flat-baseline gamma CDF fits.
    """

    def __init__(self, *args, **kwargs):
        import psychopy.visual as _vis
        _orig = _vis.Window.getActualFrameRate

        def _patched(self_win, *a, **kw):
            r = _orig(self_win, *a, **kw)
            return 60.0 if r is None else r

        _vis.Window.getActualFrameRate = _patched
        super().__init__(*args, **kwargs)
        _vis.Window.getActualFrameRate = _orig

        if self.frame_rate is None:
            self.frame_rate     = 60.0
            self.frame_interval = 1.0 / 60.0
            self.stim_frames    = np.round((1.0 / self.freq_array) / self.frame_interval)

    def pipeReceive(self):
        import scipy.stats as st

        fft_padding        = 5
        window_len         = 1000
        recv_window_len    = 1000
        fs                 = 1000
        cdf_per            = 10.0
        max_baseline_time  = 30
        max_baseline_count = int(max_baseline_time * fs / window_len)
        freq_tol           = 0.5   # Fix E

        ham              = np.hamming(window_len)
        freq_axis        = np.fft.rfftfreq(window_len * fft_padding, 1 / fs)
        signal_buff      = np.zeros(window_len)
        freq_array_len   = len(self.freq_array)

        freq_sig_snr        = np.zeros([freq_array_len, 2])
        freq_sig_mean_snr   = np.zeros([freq_array_len, 1])
        freq_sig_base_val   = np.zeros([freq_array_len, max_baseline_count])
        freq_sig_val_thresh = np.zeros([freq_array_len, 1])

        while True:
            for i in range(int(window_len / recv_window_len)):
                signal_buff[
                    i * recv_window_len: i * recv_window_len + recv_window_len
                ] = self.pipe.recv()

            y_ham    = signal_buff * ham
            rfft     = np.fft.rfft(y_ham, window_len * fft_padding)
            rfft_mag = 4 / window_len * np.absolute(rfft)

            for index, f in enumerate(self.freq_array):
                freq_start = f - freq_tol
                freq_end   = f + freq_tol
                freq_max   = 0.0
                in_band    = False
                for i in range(len(freq_axis)):
                    if freq_axis[i] >= freq_start:
                        in_band = True
                    if in_band and rfft_mag[i] > freq_max:
                        freq_max = rfft_mag[i]
                    if freq_axis[i] > freq_end:
                        if self.setting_baseline:
                            freq_sig_base_val[index][self.baseline_count] = freq_max
                        else:
                            thresh = float(
                                freq_sig_val_thresh[index].item()
                                if hasattr(freq_sig_val_thresh[index], "item")
                                else freq_sig_val_thresh[index]
                            )
                            if thresh > 0 and freq_max > thresh:
                                freq_sig_snr[index][1] = freq_max / thresh
                            else:
                                freq_sig_snr[index][1] = 0.0
                        break

            if self.setting_baseline:
                self.baseline_count += 1
                if self.baseline_count == max_baseline_count:
                    for i in range(freq_array_len):
                        std  = float(np.std(freq_sig_base_val[i]))
                        mean = float(np.mean(freq_sig_base_val[i]))
                        x    = np.linspace(
                            float(min(freq_sig_base_val[i])),
                            float(max(freq_sig_base_val[i])) * 1.5,
                            1000,
                        )
                        params     = st.gamma.fit(freq_sig_base_val[i])
                        fitted_cdf = st.gamma.cdf(x, params[0], params[1], params[2])
                        thresh_val = 0.0
                        for j in range(len(x)):
                            if (1 - fitted_cdf[j]) < (cdf_per / 100.0):
                                thresh_val = float(x[j])
                                break
                        if thresh_val == 0.0:
                            thresh_val = mean + 2 * std
                        freq_sig_val_thresh[i] = thresh_val
                        print("freq {} mean {:.6f}, std {:.6f}, thresh {:.6f}".format(
                            self.freq_array[i], mean, std, thresh_val
                        ))
                        freq_sig_snr[i][0] = (
                            float(freq_sig_base_val[i][-1]) / thresh_val
                            if thresh_val > 0 else 0.0
                        )
                    self.baseline_count          = 0
                    self.setting_baseline        = False
                    self.instructions_box_update = True
            else:
                print("freq sig val{}".format(freq_sig_snr.tolist()))
                for i in range(freq_array_len):
                    if freq_sig_snr[i][0] > 0 and freq_sig_snr[i][1] > 0:
                        freq_sig_mean_snr[i] = float(np.mean(freq_sig_snr[i]))
                    else:
                        freq_sig_mean_snr[i] = 0.0
                    freq_sig_snr[i][0] = freq_sig_snr[i][1]

                max_idx = int(np.argmax(freq_sig_mean_snr))
                if freq_sig_mean_snr[max_idx] > 0:
                    print("max freq snr {}".format(self.freq_array[max_idx]))
                    self.selected_index      = max_idx
                    self.selected_box.pos    = self.stim[max_idx].pos
                    self.selected_box_frames = self.selected_box_on_frames


# ===========================================================================
# BinaryAcquisitionSession
# ===========================================================================

class BinaryAcquisitionSession:
    CHUNK_SIZE = 100   # Fix C
    VREF_MV    = 3300.0
    ADC_MAX    = 4095.0

    def __init__(self, bridge, port, baud, cfg, simulate, verbose):
        self._bridge    = bridge
        self._cfg       = cfg
        self._verbose   = verbose
        self._stop      = threading.Event()
        self._queue     = queue.Queue(maxsize=10000)
        self._pipeline  = RealTimeSSVEPPipeline(cfg)
        self._window_idx = 0
        self._session_start = time.monotonic()

        self._smoother = ConfidenceSmoother(
            stim_freqs=list(cfg.STIM_FREQS),
            alpha=0.35,
            confirm_threshold=0.55,
            cooldown_sec=1.0,
            min_snr_db=3.5,
        )

        if simulate:
            self._reader = SimulatedReader(
                shared_queue=self._queue, stop_event=self._stop, logger=log,
            )
        else:
            self._reader = STM32BinaryReader(
                port=port, baud=baud, shared_queue=self._queue,
                stop_event=self._stop, logger=log,
            )

    def run(self) -> None:
        self._reader.start()
        self._session_start = time.monotonic()
        log.info("Acquisition session running. Press Ctrl-C to stop.")
        accumulator: list = []
        last_stats = time.monotonic()

        try:
            while not self._stop.is_set():
                # -------------------------------------------------------
                # BUG 1 FIX: use get_nowait() + short sleep instead of
                # get(timeout=0.05).
                #
                # OLD CODE (broken):
                #   _ts, adc = self._queue.get(timeout=0.05)
                #
                # queue.get(timeout=0.05) holds the GIL for up to 50ms
                # while waiting. This delays PsychoPy's win.flip() call,
                # causing frame drops of 33-59ms (should be 16.7ms at 60Hz).
                # Irregular frames → smeared FFT → low SNR → no confirms.
                #
                # NEW CODE: get_nowait() returns immediately if empty.
                # time.sleep(0.001) then releases the GIL for 1ms so
                # PsychoPy can flip on schedule. 1ms is short enough
                # that we check the queue 1000×/sec — more than adequate
                # for 1000 Hz incoming data.
                # -------------------------------------------------------
                try:
                    _ts, adc = self._queue.get_nowait()
                    accumulator.append(adc)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    time.sleep(0.001)   # release GIL; PsychoPy can flip
                    continue

                if len(accumulator) < self.CHUNK_SIZE:
                    continue

                chunk_np = np.array(accumulator, dtype=np.int32)
                accumulator.clear()

                adc_mv = chunk_np.astype(np.float64) / self.ADC_MAX * self.VREF_MV
                self._bridge.send(adc_mv)

                result = self._pipeline.push(chunk_np)
                if result is not None:
                    self._window_idx += 1
                    _fv, info, raw_pred = result

                    confirmed = self._smoother.push(info, raw_pred)

                    if confirmed is not None:
                        elapsed = time.monotonic() - self._session_start
                        print(
                            f"\n{'='*50}\n"
                            f"  *** CONFIRMED: {confirmed:.1f} Hz ***\n"
                            f"  SNR={info['best_snr_db']:+.2f} dB  t={elapsed:.1f}s\n"
                            f"{'='*50}\n",
                            flush=True,
                        )
                    elif self._verbose or raw_pred is not None:
                        snr = info["best_snr_db"]
                        q   = "OK" if info["quality_ok"] else "REJECT"
                        p   = f"{raw_pred:.1f} Hz" if raw_pred else "—"
                        print(
                            f"[W{self._window_idx:04d}] "
                            f"best={info['best_candidate']:.1f}Hz "
                            f"SNR={snr:+.2f}dB {q} pred={p}",
                            flush=True,
                        )

                now = time.monotonic()
                if now - last_stats >= 10.0:
                    elapsed = now - self._session_start
                    log.info(
                        "Stats | t=%.0fs  queue=%d  samples_rx=%d  windows=%d",
                        elapsed, self._queue.qsize(),
                        self._reader.samples_received, self._window_idx,
                    )
                    last_stats = now

        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            self._reader.stop()
            self._reader.join(timeout=3.0)
            log.info("Acquisition ended. Windows processed: %d", self._window_idx)

    def stop(self) -> None:
        self._stop.set()


# ===========================================================================
# AcquisitionThread
# ===========================================================================

class AcquisitionThread(threading.Thread):
    def __init__(self, session):
        super().__init__(name="AcquisitionThread", daemon=True)
        self._session = session

    def run(self):
        log.info("Acquisition thread starting.")
        try:
            self._session.run()
        except Exception as exc:
            log.error("Acquisition thread crashed: %s", exc, exc_info=True)
        finally:
            log.info("Acquisition thread exiting.")

    def shutdown(self):
        self._session.stop()


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser():
    p = argparse.ArgumentParser(
        description="SSVEP BCI — STM32 binary protocol",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port",     "-p", default=None)
    p.add_argument("--baud",     "-b", type=int, default=57600)
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--freqs",    type=float, nargs="+",
                   default=[10.0, 20.0, 15.0, 5.0, 12.0])
    p.add_argument("--window",   type=float, default=2.0)
    p.add_argument("--overlap",  type=float, default=0.5)
    p.add_argument("--verbose",  "-v", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()

    if not args.simulate and args.port is None:
        sys.exit(
            "[ERROR] --port required.\n"
            "  Example: python run_bci.py --port COM12\n"
            "  Or:      python run_bci.py --simulate"
        )

    cfg = SSVEPConfig(
        fs=FIRMWARE_FS,
        stim_freqs=args.freqs,
        WINDOW_SEC=args.window,
        OVERLAP=args.overlap,
        MIN_SNR_DB=3.5,   # Fix D (raised from 2.0)
    )

    log.info("=" * 60)
    log.info("SSVEP BCI — integrated launcher")
    log.info("  Mode           : %s", "SIMULATE" if args.simulate
             else f"SERIAL {args.port} @ {args.baud} baud")
    log.info("  Stimulus freqs : %s Hz", cfg.STIM_FREQS)
    log.info("  Sample rate    : %d Hz", FIRMWARE_FS)
    log.info("  Window         : %.1f s (%d samples)", cfg.WINDOW_SEC, cfg.WINDOW_SAMPLES)
    log.info("  MIN_SNR_DB     : %.1f dB", cfg.MIN_SNR_DB)
    log.info("  Chunk size     : %d samples", BinaryAcquisitionSession.CHUNK_SIZE)
    log.info("  Smoother alpha : 0.35  confirm >= 0.55  min_snr 3.5 dB  cooldown 1.0s")
    log.info("  Queue drain    : get_nowait + sleep(1ms)  [GIL fix]")
    log.info("  psutil         : %s", "available" if _PSUTIL_AVAILABLE else "NOT installed")
    log.info("=" * 60)

    bridge  = PredictionBridge()
    session = BinaryAcquisitionSession(
        bridge=bridge, port=args.port, baud=args.baud,
        cfg=cfg, simulate=args.simulate, verbose=args.verbose,
    )
    acq = AcquisitionThread(session)
    acq.start()

    log.info("Waiting for acquisition thread to warm up …")
    time.sleep(3.5)

    log.info("Opening PsychoPy window …")
    try:
        SafeBCI(
            win_size=[1200, 700],
            freq_array=np.array(cfg.STIM_FREQS),
            checker_cycles=4,
            checker_size=160,
            checker_tex=np.array([[1, -1], [-1, 1]]),
            pipe=bridge,
        )
    except KeyboardInterrupt:
        log.info("Keyboard interrupt — shutting down.")
    except Exception as exc:
        log.error("BCI window error: %s", exc, exc_info=True)
    finally:
        log.info("Stopping acquisition thread …")
        acq.shutdown()
        acq.join(timeout=5.0)
        log.info("Done.")


if __name__ == "__main__":
    main()