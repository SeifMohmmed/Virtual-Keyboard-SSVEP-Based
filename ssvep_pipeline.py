"""
SSVEP EEG Preprocessing & Feature Extraction Pipeline
======================================================
Hardware : BioAmp EXG + STM32 NUCLEO-F303K8 (12-bit ADC)
Target   : Real-time SSVEP speller, stimuli at 8–15 Hz
Author   : Pipeline design for single-channel occipital (Oz) EEG

Dependencies: numpy, scipy
Usage example at the bottom of this file.
"""

import numpy as np
from scipy import signal
from numpy.linalg import svd
from collections import deque
from typing import List, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

class SSVEPConfig:
    """Central configuration. Adjust to match your hardware and stimuli."""

    # --- Acquisition ---
    FS: int = 500
    ADC_BITS: int = 12
    VREF: float = 3.3

    # --- Windowing ---
    WINDOW_SEC: float = 2.0
    OVERLAP: float = 0.50

    # --- Filter parameters ---
    HP_CUTOFF: float = 0.5
    NOTCH_FREQ: float = 50.0
    NOTCH_Q: float = 30.0
    NOTCH_HARMONIC: bool = True
    BP_LOW: float = 6.0
    BP_HIGH: float = 30.0
    BP_ORDER: int = 4

    # --- Feature extraction ---
    STIM_FREQS: List[float] = None
    N_HARMONICS: int = 3
    BAND_HALF_BW: float = 0.5
    SNR_NOISE_OFFSET: float = 1.5
    SNR_NOISE_BW: float = 1.0

    # Harmonic weights: w[0]=fundamental, w[1]=2f, w[2]=3f
    # PHASE 3 FIX: now mutable per-instance (adaptive weights update these)
    HARMONIC_WEIGHTS: List[float] = None

    # --- Quality thresholds ---
    ENTROPY_REJECT_THRESHOLD: float = 0.92
    MIN_SNR_DB: float = 0.0

    def __init__(self,
                 fs: int = 500,
                 stim_freqs: Optional[List[float]] = None,
                 **kwargs):
        self.FS = fs
        self.STIM_FREQS = stim_freqs if stim_freqs is not None else [8.0, 10.0, 12.0, 15.0]
        # PHASE 3 FIX: always create a fresh list per instance so adaptive
        # updates on one pipeline never bleed into another
        self.HARMONIC_WEIGHTS = [1.0, 0.5, 0.25]
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @property
    def WINDOW_SAMPLES(self) -> int:
        return int(self.WINDOW_SEC * self.FS)

    @property
    def STEP_SAMPLES(self) -> int:
        return int(self.WINDOW_SAMPLES * (1.0 - self.OVERLAP))

    @property
    def FEATURE_DIM(self) -> int:
        n = len(self.STIM_FREQS)
        return n * self.N_HARMONICS + n * (self.N_HARMONICS - 1) + n + 1 + 1


# =============================================================================
# STAGE 1: PREPROCESSING
# =============================================================================

class Preprocessor:
    """
    Stateful preprocessing pipeline. Designed for real-time streaming:
    filter state (zi) is preserved across chunks so filtering is continuous.
    Call reset() when starting a new trial or session.
    """

    def __init__(self, cfg: SSVEPConfig):
        self.cfg = cfg
        self._build_filters()
        self.reset()
        self._initialized = False

    def _build_filters(self):
        cfg = self.cfg
        fs = cfg.FS

        self._hp_sos = signal.butter(
            1, cfg.HP_CUTOFF, btype='highpass', fs=fs, output='sos'
        )

        notch_freqs = [cfg.NOTCH_FREQ]
        if cfg.NOTCH_HARMONIC and cfg.NOTCH_FREQ * 2 < fs / 2:
            notch_freqs.append(cfg.NOTCH_FREQ * 2)

        self._notch_sos_list = []
        for f in notch_freqs:
            b, a = signal.iirnotch(f, cfg.NOTCH_Q, fs=fs)
            sos = signal.tf2sos(b, a)
            self._notch_sos_list.append(sos)

        self._bp_sos = signal.butter(
            cfg.BP_ORDER,
            [cfg.BP_LOW, cfg.BP_HIGH],
            btype='bandpass',
            fs=fs,
            output='sos'
        )

    def reset(self):
        """Reset all filter states. Call at the start of each new session."""
        def _zi(sos):
            return signal.sosfilt_zi(sos) * 0.0

        self._hp_zi    = _zi(self._hp_sos)
        self._notch_zi = [_zi(sos) for sos in self._notch_sos_list]
        self._bp_zi    = _zi(self._bp_sos)
        self._initialized = False

    def initialize_from_first_chunk(self, raw_chunk: np.ndarray) -> None:
        """
        Warm-start all filter initial conditions using the DC level of the
        first real chunk. Eliminates startup transient artifacts.
        """
        x = self.adc_to_mv(raw_chunk)
        dc_level = float(np.mean(x))

        self._hp_zi = signal.sosfilt_zi(self._hp_sos) * dc_level
        self._notch_zi = [
            signal.sosfilt_zi(sos) * dc_level
            for sos in self._notch_sos_list
        ]
        # Bandpass DC steady-state is always 0 — leave at zero
        self._bp_zi = signal.sosfilt_zi(self._bp_sos) * 0.0
        self._initialized = True

    def adc_to_mv(self, raw: np.ndarray) -> np.ndarray:
        vref_mv = self.cfg.VREF * 1000.0
        return (raw.astype(np.float64) / (2**self.cfg.ADC_BITS - 1)) * vref_mv

    def remove_dc(self, x: np.ndarray) -> np.ndarray:
        y, self._hp_zi = signal.sosfilt(self._hp_sos, x, zi=self._hp_zi)
        return y

    def notch_filter(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        for i, sos in enumerate(self._notch_sos_list):
            y, self._notch_zi[i] = signal.sosfilt(sos, y, zi=self._notch_zi[i])
        return y

    def bandpass_filter(self, x: np.ndarray) -> np.ndarray:
        y, self._bp_zi = signal.sosfilt(self._bp_sos, x, zi=self._bp_zi)
        return y

    @staticmethod
    def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mu  = np.mean(x)
        std = np.std(x)
        return (x - mu) / (std + eps)

    def process_chunk(self, raw_chunk: np.ndarray) -> np.ndarray:
        """
        Process a streaming chunk. Auto warm-starts on first call.
        Chain: ADC scale → DC removal → notch → bandpass.
        """
        if not self._initialized:
            self.initialize_from_first_chunk(raw_chunk)
        x = self.adc_to_mv(raw_chunk)
        x = self.remove_dc(x)
        x = self.notch_filter(x)
        x = self.bandpass_filter(x)
        return x

    def process_window(self, raw_window: np.ndarray) -> np.ndarray:
        x = self.process_chunk(raw_window)
        x = self.normalize(x)
        return x


# =============================================================================
# STAGE 2: WINDOWING — REAL-TIME RING BUFFER
# =============================================================================

class StreamingWindowBuffer:
    def __init__(self, cfg: SSVEPConfig):
        self.cfg = cfg
        self._buffer = deque(maxlen=cfg.WINDOW_SAMPLES)
        self._sample_count = 0

    def push(self, samples: np.ndarray):
        for s in samples:
            self._buffer.append(s)
            self._sample_count += 1

    def get_latest_window(self) -> Optional[np.ndarray]:
        if len(self._buffer) == self.cfg.WINDOW_SAMPLES:
            return np.array(self._buffer)
        return None

    @property
    def fill_ratio(self) -> float:
        return len(self._buffer) / self.cfg.WINDOW_SAMPLES

    def get_latest_chunk(self, n: int = 1) -> np.ndarray:
        arr = list(self._buffer)
        return np.array(arr[-n:]) if len(arr) >= n else np.array([])


class PollingWindowBuffer:
    """
    Zero-copy circular buffer — replaces np.roll with a double-length
    array and a write pointer. O(chunk) per add instead of O(window).
    """

    def __init__(self, cfg: SSVEPConfig):
        self.cfg = cfg
        ws = cfg.WINDOW_SAMPLES
        self._buf          = np.zeros(ws * 2, dtype=np.float64)
        self._write_pos    = 0
        self._filled       = 0
        self._step_counter = 0
        self._window_ready = False

    def add(self, chunk: np.ndarray):
        n  = len(chunk)
        ws = self.cfg.WINDOW_SAMPLES
        end = self._write_pos + n
        if end <= ws * 2:
            self._buf[self._write_pos:end] = chunk
        else:
            split = ws * 2 - self._write_pos
            self._buf[self._write_pos:] = chunk[:split]
            self._buf[:n - split]       = chunk[split:]
        self._write_pos = end % (ws * 2)
        self._filled    = min(self._filled + n, ws)
        self._step_counter += n
        if self._step_counter >= self.cfg.STEP_SAMPLES and self._filled >= ws:
            self._step_counter = 0
            self._window_ready = True

    def ready(self) -> bool:
        return self._window_ready

    def get_window(self) -> np.ndarray:
        self._window_ready = False
        ws    = self.cfg.WINDOW_SAMPLES
        start = self._write_pos
        if start + ws <= ws * 2:
            return self._buf[start:start + ws].copy()
        split = ws * 2 - start
        return np.concatenate([self._buf[start:], self._buf[:ws - split]])


# =============================================================================
# STAGE 3: FEATURE EXTRACTION  (Phase 3 fixes applied)
# =============================================================================

# ---------------------------------------------------------------------------
# PHASE 3 FIX 3 — CCA scorer (module-level helper, stateless)
# ---------------------------------------------------------------------------
def cca_score(window: np.ndarray, stim_freq: float, fs: int,
              n_harmonics: int = 3) -> float:
    """
    Canonical Correlation Analysis score between a single-channel EEG
    window and a set of sinusoidal reference signals at stim_freq.

    Returns the maximum canonical correlation coefficient in [0, 1].
    Higher = stronger match to the stimulus frequency.

    Why CCA works here
    ------------------
    CCA finds the linear combination of the reference signals (sin/cos at
    f, 2f, 3f ...) that is maximally correlated with the EEG window.
    For a single EEG channel this reduces to the largest singular value of
    Qx.T @ Qy, where Qx and Qy are the orthonormal bases of the (mean-
    centred) EEG and reference matrices respectively.
    """
    t = np.arange(len(window)) / fs
    Y_cols = []
    for k in range(1, n_harmonics + 1):
        Y_cols.append(np.sin(2 * np.pi * k * stim_freq * t))
        Y_cols.append(np.cos(2 * np.pi * k * stim_freq * t))
    Y = np.column_stack(Y_cols)           # (N, 2*n_harmonics)
    X = window.reshape(-1, 1)             # (N, 1)

    Qx, _ = np.linalg.qr(X - X.mean(0))
    Qy, _ = np.linalg.qr(Y - Y.mean(0))
    _, S, _ = svd(Qx.T @ Qy, full_matrices=False)
    return float(S[0])                    # max canonical correlation


class SSVEPFeatureExtractor:
    """
    Extract SSVEP-specific features from a single preprocessed, normalised
    window. All methods are stateless — safe to call from any thread.

    Phase 3 changes
    ---------------
    1. nperseg = WINDOW_SAMPLES   (full-window Welch, 0.5 Hz resolution)
    2. update_harmonic_weights()  (adaptive EMA weight tuning)
    3. cca_scores in extract()    (CCA alongside Welch SNR)
    4. classify() fuses SNR + CCA (60 / 40 blend)
    """

    def __init__(self, cfg: SSVEPConfig):
        self.cfg = cfg
        self._build_frequency_axes()
        # Counter for adaptive harmonic weight updates (every 10 windows)
        self._window_count = 0

    def _build_frequency_axes(self):
        cfg = self.cfg
        # PHASE 3 FIX 1: use full window length for maximum frequency resolution
        # At 1000 Hz, 2 s window → nperseg=2000 → freq_res = 0.5 Hz
        # Stimuli at 8 Hz and 10 Hz are now 4 bins apart — clearly separable
        self._nperseg  = cfg.WINDOW_SAMPLES          # was: min(256, WINDOW_SAMPLES)
        self._freq_res = cfg.FS / self._nperseg

    # ------------------------------------------------------------------
    # PHASE 3 FIX 2 — adaptive harmonic weights
    # ------------------------------------------------------------------
    def update_harmonic_weights(self, snr_per_harmonic: list) -> None:
        """
        Update cfg.HARMONIC_WEIGHTS using an exponential moving average
        driven by the observed per-harmonic SNR values.

        Call after every window (the method only mutates weights every
        10 windows internally to avoid over-fitting to noise).

        Parameters
        ----------
        snr_per_harmonic : [snr_f, snr_2f, snr_3f, ...]  (dB, one entry
                           per harmonic up to N_HARMONICS)
        """
        self._window_count += 1
        if self._window_count % 10 != 0:
            return

        # Convert dB SNR → linear power proxy; clamp negatives to zero
        powers = np.array([max(0.0, s) for s in snr_per_harmonic], dtype=float)
        powers += 1e-9                    # avoid all-zero division
        normalised = powers / powers.sum()

        alpha = 0.1                       # EMA smoothing — slow, stable
        weights = self.cfg.HARMONIC_WEIGHTS
        for k in range(len(weights)):
            if k < len(normalised):
                weights[k] = (1.0 - alpha) * weights[k] + alpha * float(normalised[k])

    # ------------------------------------------------------------------
    # F1: Welch PSD
    # ------------------------------------------------------------------
    def compute_psd(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch method.
        nperseg = WINDOW_SAMPLES → maximum frequency resolution.
        noverlap = nperseg // 2  → standard 50 % Welch overlap.
        """
        freqs, psd = signal.welch(
            window,
            fs=self.cfg.FS,
            window='hann',
            nperseg=self._nperseg,
            noverlap=self._nperseg // 2,
            scaling='density',
            detrend='constant'
        )
        return freqs, psd

    # ------------------------------------------------------------------
    # F2: Band Power
    # ------------------------------------------------------------------
    def band_power(self,
                   freqs: np.ndarray,
                   psd: np.ndarray,
                   center_freq: float,
                   half_bw: Optional[float] = None) -> float:
        if half_bw is None:
            half_bw = self.cfg.BAND_HALF_BW
        lo   = center_freq - half_bw
        hi   = center_freq + half_bw
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))

    # ------------------------------------------------------------------
    # F3: SNR per target frequency
    # ------------------------------------------------------------------
    def compute_snr(self,
                    freqs: np.ndarray,
                    psd: np.ndarray,
                    target_freq: float) -> float:
        cfg        = self.cfg
        p_sig      = self.band_power(freqs, psd, target_freq, cfg.BAND_HALF_BW)
        lo_center  = target_freq - cfg.SNR_NOISE_OFFSET - cfg.SNR_NOISE_BW / 2
        hi_center  = target_freq + cfg.SNR_NOISE_OFFSET + cfg.SNR_NOISE_BW / 2
        p_noise_lo = self.band_power(freqs, psd, lo_center, cfg.SNR_NOISE_BW / 2)
        p_noise_hi = self.band_power(freqs, psd, hi_center, cfg.SNR_NOISE_BW / 2)
        p_noise    = (p_noise_lo + p_noise_hi) / 2.0 + 1e-30
        return float(10.0 * np.log10(p_sig / p_noise + 1e-30))

    # ------------------------------------------------------------------
    # F4: Harmonic score
    # ------------------------------------------------------------------
    def harmonic_score(self,
                       freqs: np.ndarray,
                       psd: np.ndarray,
                       stim_freq: float) -> float:
        cfg   = self.cfg
        nyq   = cfg.FS / 2.0
        score = 0.0
        for k in range(1, cfg.N_HARMONICS + 1):
            hf = k * stim_freq
            if hf > nyq or hf > cfg.BP_HIGH:
                break
            w      = cfg.HARMONIC_WEIGHTS[k - 1] if k - 1 < len(cfg.HARMONIC_WEIGHTS) else 0.1
            score += w * self.band_power(freqs, psd, hf)
        return float(score)

    # ------------------------------------------------------------------
    # F5: Spectral entropy
    # ------------------------------------------------------------------
    def spectral_entropy(self,
                         freqs: np.ndarray,
                         psd: np.ndarray,
                         f_low: Optional[float] = None,
                         f_high: Optional[float] = None) -> float:
        f_low  = f_low  or self.cfg.BP_LOW
        f_high = f_high or self.cfg.BP_HIGH
        mask   = (freqs >= f_low) & (freqs <= f_high)
        p      = psd[mask]
        p_norm = p / (np.sum(p) + 1e-30)
        H      = -np.sum(p_norm * np.log(p_norm + 1e-30))
        H_max  = np.log(np.sum(mask))
        return float(H / (H_max + 1e-30))

    # ------------------------------------------------------------------
    # Peak frequency
    # ------------------------------------------------------------------
    def peak_frequency(self,
                       freqs: np.ndarray,
                       psd: np.ndarray,
                       f_low: Optional[float] = None,
                       f_high: Optional[float] = None) -> float:
        f_low  = f_low  or self.cfg.BP_LOW
        f_high = f_high or self.cfg.BP_HIGH
        mask   = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(mask):
            return 0.0
        return float(freqs[mask][np.argmax(psd[mask])])

    # ------------------------------------------------------------------
    # Full feature vector  (PHASE 3: now includes CCA scores)
    # ------------------------------------------------------------------
    def extract(self,
                window: np.ndarray,
                normalize: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Full feature extraction. Now computes both Welch-SNR features and
        CCA scores for every stimulus frequency.

        New keys in info dict
        ---------------------
        'cca_scores'   : list[float]  — CCA coefficient per stimulus [0,1]
        'combined_scores' : list[float] — 0.6*SNR_norm + 0.4*CCA per stimulus
        """
        cfg = self.cfg

        if normalize:
            window = Preprocessor.normalize(window)

        freqs, psd = self.compute_psd(window)

        bp_fundamental  = []
        bp_harmonics    = []
        snr_fundamental = []
        snr_harmonics   = []
        harm_scores     = []

        for f in cfg.STIM_FREQS:
            bp_fundamental.append(self.band_power(freqs, psd, f))
            snr_fundamental.append(self.compute_snr(freqs, psd, f))
            harm_scores.append(self.harmonic_score(freqs, psd, f))

            h_bps, h_snrs = [], []
            for k in range(2, cfg.N_HARMONICS + 1):
                hf = k * f
                if hf < cfg.FS / 2 and hf <= cfg.BP_HIGH:
                    h_bps.append(self.band_power(freqs, psd, hf))
                    h_snrs.append(self.compute_snr(freqs, psd, hf))
                else:
                    h_bps.append(0.0)
                    h_snrs.append(0.0)
            bp_harmonics.append(h_bps)
            snr_harmonics.append(h_snrs)

        entropy   = self.spectral_entropy(freqs, psd)
        peak_freq = self.peak_frequency(freqs, psd)

        # PHASE 3 FIX 3 — compute CCA score for every stimulus frequency
        cca_scores = [
            cca_score(window, f, cfg.FS, cfg.N_HARMONICS)
            for f in cfg.STIM_FREQS
        ]

        # Fused score: normalise SNR to [0,1] then blend 60/40 with CCA
        snr_arr = np.array(snr_fundamental)
        snr_min, snr_max = snr_arr.min(), snr_arr.max()
        snr_range = snr_max - snr_min + 1e-9
        snr_norm  = (snr_arr - snr_min) / snr_range          # [0, 1]
        cca_arr   = np.array(cca_scores)
        combined  = (0.9 * snr_norm + 0.1 * cca_arr).tolist()

        # PHASE 3 FIX 2 — adaptive harmonic weight update
        # Build per-harmonic SNR list for the best candidate stimulus
        best_snr_idx       = int(np.argmax(snr_fundamental))
        best_f             = cfg.STIM_FREQS[best_snr_idx]
        snr_harmonics_best = [snr_fundamental[best_snr_idx]] + \
                             snr_harmonics[best_snr_idx]
        self.update_harmonic_weights(snr_harmonics_best)

        fv = np.array(
            bp_fundamental
            + [v for row in bp_harmonics  for v in row]
            + snr_fundamental
            + [v for row in snr_harmonics for v in row]
            + harm_scores
            + [entropy, peak_freq]
        )

        best_idx  = int(np.argmax(snr_fundamental))
        best_freq = cfg.STIM_FREQS[best_idx]
        best_snr  = snr_fundamental[best_idx]
        quality_ok = (entropy < cfg.ENTROPY_REJECT_THRESHOLD and
                      best_snr > cfg.MIN_SNR_DB)

        info = {
            "stim_freqs":        cfg.STIM_FREQS,
            "bp_fundamental":    bp_fundamental,
            "bp_harmonics":      bp_harmonics,
            "snr_fundamental":   snr_fundamental,
            "snr_harmonics":     snr_harmonics,
            "harmonic_scores":   harm_scores,
            "cca_scores":        cca_scores,        # NEW
            "combined_scores":   combined,           # NEW
            "spectral_entropy":  entropy,
            "peak_frequency":    peak_freq,
            "best_candidate":    best_freq,
            "best_snr_db":       best_snr,
            "quality_ok":        quality_ok,
        }

        return fv, info

    # ------------------------------------------------------------------
    # classify — PHASE 3: fuses SNR + CCA via combined_scores
    # ------------------------------------------------------------------
    def classify(self,
                 feature_vector: np.ndarray,
                 info: dict,
                 min_snr_db: float = 2.0) -> Optional[float]:
        """
        Rule-based classifier using the fused SNR+CCA score.

        Decision rule
        -------------
        1. Reject the window if quality_ok is False.
        2. Pick the stimulus with the highest combined_score.
        3. Accept only if its raw SNR is also above min_snr_db.
           (CCA alone can be fooled by correlated noise; the SNR gate
           prevents spurious detections when the signal is absent.)
        """
        if not info["quality_ok"]:
            return None

        combined  = np.array(info["combined_scores"])
        snr_scores = info["snr_fundamental"]
        best_idx  = int(np.argmax(combined))

        if snr_scores[best_idx] >= min_snr_db:
            return self.cfg.STIM_FREQS[best_idx]
        return None


# =============================================================================
# REAL-TIME PIPELINE ORCHESTRATOR
# =============================================================================

class RealTimeSSVEPPipeline:
    """
    Ties everything together for a real-time streaming scenario.

    Typical usage:
        pipeline = RealTimeSSVEPPipeline(cfg)
        for raw_chunk in serial_reader():
            result = pipeline.push(raw_chunk)
            if result is not None:
                fv, info, prediction = result
                print(f"Detected: {prediction} Hz  SNR: {info['best_snr_db']:.1f} dB")
    """

    def __init__(self, cfg: SSVEPConfig):
        self.cfg          = cfg
        self.preprocessor = Preprocessor(cfg)
        self.buffer       = PollingWindowBuffer(cfg)
        self.extractor    = SSVEPFeatureExtractor(cfg)

    def push(self, raw_chunk: np.ndarray) -> Optional[Tuple]:
        filtered = self.preprocessor.process_chunk(raw_chunk)
        self.buffer.add(filtered)

        if not self.buffer.ready():
            return None

        window      = self.buffer.get_window()
        norm_window = Preprocessor.normalize(window)
        fv, info    = self.extractor.extract(norm_window, normalize=False)
        prediction  = self.extractor.classify(fv, info)

        return fv, info, prediction

    def reset(self):
        self.preprocessor.reset()
        self.buffer = PollingWindowBuffer(self.cfg)


# =============================================================================
# CSV PARSER
# =============================================================================

def parse_stm32_csv(line: str) -> Tuple[Optional[float], Optional[int]]:
    try:
        parts = line.strip().split(',')
        if len(parts) < 2:
            return None, None
        return float(parts[0]) / 1000.0, int(parts[1])
    except (ValueError, IndexError):
        return None, None


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import time

    print("=== SSVEP Pipeline Demo (Phase 3) ===\n")

    cfg = SSVEPConfig(
        fs=1000,
        stim_freqs=[8.0, 10.0, 12.0, 15.0],
        WINDOW_SEC=2.0,
        OVERLAP=0.50,
    )
    print(f"Window      : {cfg.WINDOW_SEC} s ({cfg.WINDOW_SAMPLES} samples)")
    print(f"Step        : {cfg.STEP_SAMPLES} samples")
    print(f"nperseg     : {cfg.WINDOW_SAMPLES}  →  freq_res = {cfg.FS/cfg.WINDOW_SAMPLES:.2f} Hz")
    print(f"Stimuli     : {cfg.STIM_FREQS} Hz\n")

    rng = np.random.default_rng(42)
    fs  = cfg.FS
    t   = np.arange(0, 10, 1.0 / fs)
    target_freq = 10.0
    clean_uv = (
        50.0 * np.sin(2 * np.pi * target_freq * t)
        + 20.0 * np.sin(2 * np.pi * 2 * target_freq * t)
        + 5.0  * np.sin(2 * np.pi * 3 * target_freq * t)
    )
    noise      = rng.normal(0, 30, len(t))
    powerline  = 200.0 * np.sin(2 * np.pi * 50.0 * t)
    dc_drift   = 100.0 + 50.0 * np.sin(2 * np.pi * 0.05 * t)
    signal_mv  = (clean_uv + noise + powerline + dc_drift) / 1000.0
    vref_mv    = cfg.VREF * 1000.0
    raw_adc    = np.clip(
        np.round(signal_mv / vref_mv * (2**cfg.ADC_BITS - 1)).astype(int),
        0, 2**cfg.ADC_BITS - 1
    )

    pipeline   = RealTimeSSVEPPipeline(cfg)
    chunk_size = 50
    results    = []
    t_start    = time.time()

    for i in range(0, len(raw_adc) - chunk_size, chunk_size):
        chunk  = raw_adc[i: i + chunk_size]
        result = pipeline.push(chunk)
        if result is not None:
            fv, info, pred = result
            results.append((info, pred))

    t_elapsed = time.time() - t_start
    n_samples  = len(raw_adc)
    print(f"Processed {n_samples} samples in {t_elapsed*1000:.1f} ms  "
          f"({(n_samples/fs)/t_elapsed:.0f}x real-time)\n")

    print(f"{'Win':<5} {'Best Hz':<10} {'SNR dB':<10} {'CCA':<8} {'Combined':<10} {'Entropy':<10} {'Pred'}")
    print("-" * 65)
    for idx, (info, pred) in enumerate(results[-6:]):
        w   = len(results) - 6 + idx
        bi  = int(np.argmax(info['snr_fundamental']))
        cca = info['cca_scores'][bi]
        cmb = info['combined_scores'][bi]
        p   = f"{pred} Hz" if pred else "—"
        print(f"{w:<5} {info['best_candidate']:<10.1f} "
              f"{info['best_snr_db']:<10.2f} {cca:<8.3f} {cmb:<10.3f} "
              f"{info['spectral_entropy']:<10.3f} {p}")

    print("\nSNR + CCA breakdown (last window):")
    last = results[-1][0]
    for f, snr, cca, cmb in zip(
        cfg.STIM_FREQS,
        last['snr_fundamental'],
        last['cca_scores'],
        last['combined_scores'],
    ):
        bar = "█" * max(0, int(cmb * 20))
        print(f"  {f:5.1f} Hz  SNR={snr:+6.2f} dB  CCA={cca:.3f}  combined={cmb:.3f}  {bar}")

    print(f"\nFinal harmonic weights (adaptive): {[f'{w:.3f}' for w in cfg.HARMONIC_WEIGHTS]}")