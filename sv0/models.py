import numpy as np

class DummyModel:
    """A toy model that outputs an embedding vector per step.


    Behaves like a noisy multi-frequency oscillator in D dims.
    """
    def __init__(self, dim=16, base_freq=0.02, drift=0.0005, seed=None):
        self.dim = dim
        self.t = 0
        self.base_freq = base_freq
        self.drift = drift
        self.rng = np.random.default_rng(seed)

        # Random phase and small per-dimension detuning
        self.phase = self.rng.uniform(0, 2*np.pi, size=dim)
        self.detune = self.rng.normal(0, 0.002, size=dim)

    def __call__(self, x=None):
        self.t += 1
        freqs = self.base_freq + self.detune + self.drift * self.t
        # Multi-sine with noise
        signal = np.sin(self.phase + 2*np.pi*freqs*self.t)
        signal += 0.1 * self.rng.normal(size=self.dim)
        # small slow trend
        signal += 0.001 * self.t
        return signal.astype(np.float64)
