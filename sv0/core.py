import numpy as np
from scipy.signal import hilbert
from scipy.spatial import KDTree
from collections import deque
from .cache import BoundedLRUCache

class OptimizedSV0:
    """    Optimized Self-Violation Protocol Zero
    
    Key optimizations:
    - Hierarchical coupling instead of N^2 all-to-all
    - Sliding window FFT with overlap-add (simplified)
    - Incremental phase updates
    - Memory-bounded caching
    """
    def __init__(self, models, config=None, rng_seed=42):
        self.models = list(models)
        self.N = len(self.models)
        self.rng = np.random.default_rng(rng_seed)

        self.config = config or {
            'window_size': 128,
            'overlap': 0.5,
            'coupling_neighbors': min(5, self.N-1) if self.N>1 else 0,
            'cache_size': 1000,
            'damping_factor': 0.95,
            'phase_wrap_threshold': np.pi,
            'min_plv_threshold': 0.4,
            'max_plv_threshold': 0.8,
        }

        self.coupling_tree = self._build_coupling_hierarchy()
        self.trajectory_buffers = [deque(maxlen=self.config['window_size']) for _ in range(self.N)]
        self.fft_state = [None] * self.N
        self.freq_cache = BoundedLRUCache(self.config['cache_size'])

        # State for reporting
        self._last_sync = 0.0
        self._violation_history = []

    # ---------------- Coupling ----------------
    def _build_coupling_hierarchy(self):
        if self.N <= 5:
            m = np.ones((self.N, self.N)) - np.eye(self.N)
            return m
        positions = self.rng.normal(size=(self.N, 3))
        tree = KDTree(positions)
        k = max(1, int(self.config['coupling_neighbors']))
        coupling = np.zeros((self.N, self.N))
        for i in range(self.N):
            distances, indices = tree.query(positions[i], k=k+1)
            for j_idx, j in enumerate(indices[1:], start=1):
                # Avoid log(0) issues; scale affinity by exp(-d)
                d = float(distances[j_idx])
                coupling[i, j] = np.exp(-d)
        coupling = 0.5 * (coupling + coupling.T)
        return coupling

    # ---------------- Incremental FFT ----------------
    def process_trajectory_incremental(self, model_idx, new_embedding):
        buf = self.trajectory_buffers[model_idx]
        buf.append(np.asarray(new_embedding))
        if len(buf) < self.config['window_size']:
            return None
        trajectory = np.array(buf)  # shape [W, D]
        cache_key = hash(trajectory.tobytes()) % (2**32)
        if cache_key in self.freq_cache:
            return self.freq_cache[cache_key]
        window = np.hanning(len(trajectory))
        windowed = trajectory * window[:, None]
        if self.fft_state[model_idx] is not None:
            overlap_size = int(self.config['window_size'] * self.config['overlap'])
            freq_spectrum = self._incremental_fft(windowed, self.fft_state[model_idx], overlap_size)
        else:
            freq_spectrum = np.fft.fft(windowed, axis=0)
        self.fft_state[model_idx] = freq_spectrum
        self.freq_cache[cache_key] = freq_spectrum
        return freq_spectrum

    def _incremental_fft(self, new_data, prev_fft, overlap_size):
        alpha = 0.5
        new_fft = np.fft.fft(new_data, axis=0)
        return alpha * prev_fft + (1 - alpha) * new_fft

    # ---------------- Constraint detection ----------------
    def detect_constraint_optimized(self, freq_spectrum):
        epsilon = 1e-10
        power = (np.abs(freq_spectrum) ** 2) + epsilon
        # -- Frequency axis (normalized 0..0.5 due to real-valued signals) --
        n = freq_spectrum.shape[0]
        freq_bins = np.fft.fftfreq(n)
        mask = (np.abs(freq_bins) >= 1e-3) & (np.abs(freq_bins) <= 0.5)
        if not np.any(mask):
            return None
        semantic_power = power[mask].mean(axis=1)  # average across dims
        semantic_freqs = freq_bins[mask]
        # Smooth via moving average (edge-padded)
        smoothed = self._smooth_spectrum(semantic_power, window_size=5)
        min_idx = int(np.argmin(smoothed))
        constraint_freq = float(semantic_freqs[min_idx])
        confidence = 1.0 - (float(smoothed[min_idx]) / (float(np.mean(smoothed)) + epsilon))
        pr = float(smoothed[min_idx] / (float(np.max(smoothed)) + epsilon))
        return {'frequency': constraint_freq, 'confidence': float(np.clip(confidence, 0, 1)), 'power_ratio': pr}

    def _smooth_spectrum(self, spectrum, window_size=5):
        kernel = np.ones(window_size) / window_size
        pad = window_size // 2
        padded = np.pad(spectrum, pad, mode='edge')
        return np.convolve(padded, kernel, mode='valid')

    # ---------------- Phase locking ----------------
    def phase_lock_models_stable(self, phases, coupling_strengths):
        N = len(phases)
        phases = np.asarray(phases, dtype=float)
        new_phases = phases.copy()
        order_param = np.abs(np.mean(np.exp(1j * phases)))
        adaptive_strength = coupling_strengths * (1 - order_param * 0.5)
        for i in range(N):
            if adaptive_strength[i] < 1e-6:
                continue
            acc = 0.0
            for j in range(N):
                if i == j: 
                    continue
                w = self.coupling_tree[i, j]
                if w <= 0: 
                    continue
                diff = phases[j] - phases[i]
                diff = np.angle(np.exp(1j * diff))  # wrap to [-pi, pi]
                acc += w * np.sin(diff)
            phase_update = adaptive_strength[i] * acc / max(N, 1)
            phase_update *= self.config['damping_factor']
            new_phases[i] = (phases[i] + phase_update) % (2*np.pi)
        return new_phases, float(order_param)

    def compute_plv_stable(self, phases1, phases2):
        p1 = np.asarray(phases1, dtype=float)
        p2 = np.asarray(phases2, dtype=float)
        if p1.size == 0 or p2.size == 0:
            return 0.0
        n = min(p1.size, p2.size)
        p1 = p1[:n]; p2 = p2[:n]
        diff = p1 - p2
        plv = np.abs(np.mean(np.exp(1j*diff)))
        return float(np.clip(plv, 0, 1))

    # ---------------- Weak ties ----------------
    def detect_weak_ties_fast(self, embeddings):
        N = len(embeddings)
        if N < 2:
            return []
        points = np.array([np.ravel(e) for e in embeddings], dtype=float)
        tree = KDTree(points)
        # Heuristic thresholds from data dispersion
        col_std = tree.data.std(axis=0)
        d_min = float(np.percentile(col_std, 25))
        d_max = float(np.percentile(col_std, 75))
        weak = []
        for i in range(N):
            neighbors = tree.query_ball_point(points[i], d_max if d_max>0 else 1.0)
            for j in neighbors:
                if i >= j:
                    continue
                dist = float(np.linalg.norm(points[i] - points[j]))
                if d_min < dist < d_max and self.coupling_tree[i, j] < 0.3:
                    weak.append((i, j, dist))
        return weak

    # ---------------- Iteration ----------------
    def run_iteration_optimized(self, inputs):
        results = {'constraints': [], 'violations': [], 'emergences': [], 'synchronization': 0.0}
        freq_spectra = []
        for i, model in enumerate(self.models):
            out = model(inputs[i] if i < len(inputs) else None)
            freq = self.process_trajectory_incremental(i, out)
            if freq is not None:
                freq_spectra.append(freq)
        if len(freq_spectra) < self.N:
            return results
        constraints = [self.detect_constraint_optimized(f) for f in freq_spectra]
        results['constraints'] = [c for c in constraints if c is not None]
        current_embeddings = [buf[-1] for buf in self.trajectory_buffers]
        weak_ties = self.detect_weak_ties_fast(current_embeddings)
        if len(results['constraints']) > 0:
            phases = np.array([c['frequency'] * 2 * np.pi for c in results['constraints']], dtype=float)
            coupling_strengths = np.ones_like(phases) * 0.1
            new_phases, sync_level = self.phase_lock_models_stable(phases, coupling_strengths)
            results['synchronization'] = float(sync_level)
            self._last_sync = float(sync_level)
            for (i, j, dist) in weak_ties:
                if i < new_phases.size and j < new_phases.size:
                    plv = self.compute_plv_stable(np.array([new_phases[i]]), np.array([new_phases[j]]))
                    if (self.config['min_plv_threshold'] < plv < self.config['max_plv_threshold']):
                        evt = {'type': 'weak_tie_activation', 'nodes': (int(i), int(j)), 'plv': float(plv), 'distance': float(dist)}
                        results['violations'].append(evt)
                        self._violation_history.append(evt)
        return results

    # ---------------- Introspection for monitoring ----------------
    def get_current_sync(self):
        return float(self._last_sync)

    def get_violation_rate(self, window=100):
        if not self._violation_history:
            return 0.0
        recent = self._violation_history[-window:]
        # rate = proportion of steps with at least one violation
        # Here, approximate by counting steps that had any events;
        # we logged events flat, so we approximate via density:
        return min(1.0, len(recent) / max(1, window))
