import numpy as np
import matplotlib.pyplot as plt

# --- 1. Import The Engine (SV0) ---
from sv0.core import OptimizedSV0
from sv0.models import DummyModel
from sv0.ctn_bridge import CTNManifold

# --- 2. Import The Observer (CTN Metrics) ---
from ctn.core import lyapunov_energy, order_parameter_r
from ctn.tda import betti0_persistence

def main():
    # --- Setup ---
    N = 10
    DIM = 8
    # Create agents (noisy oscillators)
    models = [DummyModel(dim=DIM, base_freq=0.05, seed=i) for i in range(N)]
    
    # Initialize the Engine
    # We use a tighter window and cache for this responsive demo
    sv0 = OptimizedSV0(
        models=models, 
        config={'window_size': 32, 'coupling_neighbors': 4, 'cache_size': 500}
    )
    
    # Initialize the Topology Bridge
    # This represents the "constraint manifold" the agents live on
    manifold = CTNManifold(dim=DIM, lambda_decay=0.05)

    print(f"System Online: {N} Agents, {DIM} Dimensions")
    print("-" * 60)
    print(f"{'Step':<6} | {'SV0 Sync':<10} | {'CTN Lyapunov (L)':<18} | {'TDA Components'}")
    print("-" * 60)

    # Storage for plotting
    history_sync = []
    history_lyap = []

    # --- Simulation Loop ---
    for t in range(150):
        # 1. GET TOPOLOGY INPUT
        # The agents receive the current state of the manifold
        topo_input = manifold.get_topology_embedding()
        inputs = [topo_input for _ in range(N)]

        # 2. RUN ENGINE (SV0)
        # The engine steps the models and calculates internal sync
        results = sv0.run_iteration_optimized(inputs)
        
        # 3. OBSERVE (CTN Metrics)
        # We peek inside the models to measure their 'real' stability.
        # Note: DummyModel has vector phases; we use the 0-th dimension as a proxy.
        current_phases = np.array([(m.phase[0] + m.base_freq * m.t * 2*np.pi) % (2*np.pi) for m in models])
        
        # Calculate Lyapunov Energy: L = α·Var(θ) + β·r²
        # We use a previous phase state to calculate 'delta' (velocity)
        if t > 0:
            delta_vec = (current_phases - prev_phases + np.pi) % (2*np.pi) - np.pi
        else:
            delta_vec = np.zeros_like(current_phases)
            
        L = lyapunov_energy(current_phases, delta_vec=delta_vec, alpha=1.0, beta=0.5)
        
        # Calculate Global Order Parameter (Independent check)
        r_ctn = order_parameter_r(current_phases)
        
        # 4. TDA CHECK (Topological Data Analysis)
        # Every 50 steps, check if the swarm has broken into clusters
        tda_info = ""
        if t % 50 == 0:
            # Embed phases on unit circle for TDA
            X = np.stack([np.cos(current_phases), np.sin(current_phases)], axis=1)
            tda_res = betti0_persistence(X)
            # Count how many components exist at a 'medium' distance threshold (e.g., 0.5)
            # This tells us if the group is one whole or fragmented.
            comps = [c for eps, c in zip(tda_res['epsilons'], tda_res['components']) if eps < 0.5]
            if comps:
                tda_info = f"Clusters: {min(comps)}"

        # 5. FEEDBACK
        # If SV0 detects internal constraints, we deform the manifold
        if results['constraints']:
            # Push manifold slightly along the detected frequency vector
            # (Simplified feedback logic)
            freq = results['constraints'][0]['frequency']
            feedback = np.ones(DIM) * np.sin(freq * t)
            manifold.apply_constraint_feedback(feedback, eta=0.02)

        # Store data
        sv0_sync = results.get('synchronization', 0.0)
        history_sync.append(sv0_sync)
        history_lyap.append(L)
        prev_phases = current_phases

        # Log output
        if t % 10 == 0:
            print(f"{t:<6} | {sv0_sync:<10.3f} | {L:<18.3f} | {tda_info}")

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(history_sync, label='SV0 Synchronization (Internal)')
    plt.plot(history_lyap, label='CTN Lyapunov Energy (External)', linestyle='--')
    plt.title("System Convergence: Engine vs. Observer")
    plt.xlabel("Simulation Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()