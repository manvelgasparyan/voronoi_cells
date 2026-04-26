Voronoi Partitions in Deep Neural Networks
===========================================

Experimental validation of "Deep Networks Learn Voronoi Partitions of Feature Space"

Paper Abstract
---------------
Understanding the geometric structure of neural representations is a central challenge in modern machine learning. This work shows that deep neural classifiers implicitly organize feature space through Voronoi partitions induced by class prototypes. Neural representations evolve toward centroidal Voronoi tessellations where class prototypes coincide with the centroids of their feature regions.

Repository Structure
--------------------
step1_voronoi_cells.py      - Voronoi cell visualization and purity analysis
step2_centroidal.py         - Centroidal Voronoi condition testing
step3_energy.py             - Voronoi quantization energy tracking
step4_flow.py               - Voronoi flow dynamics (Theorem 3)
step5_fixedpoint.py         - Fixed-point analysis (Theorem 5)
step6_collisions.py         - Feature collisions vs dimension sweep
step7_neural_collapse.py    - Symmetry vs separation (Theorem 7)
step8_boundaries.py         - Error concentration near Voronoi boundaries
step9_architecture.py       - Architecture and kernel effects (NTK)

Requirements
------------
pip install torch torchvision numpy matplotlib scipy tqdm

Experiments Overview
--------------------
Step 1: Voronoi Cell Visualization - Cell purity, collision mass
Step 2: Centroidal Condition - Centroidal error, gradient norm (Theorem 5)
Step 3: Quantization Energy - Voronoi energy, compactness
Step 4: Voronoi Flow Dynamics - Cosine similarity, confidence (Theorem 3)
Step 5: Fixed-Point Analysis - Stationarity, centroidal error (Theorem 5)
Step 6: Feature Collisions - Collision mass vs dimension
Step 7: Neural Collapse - ETF deviation, collision under symmetry (Theorem 7)
Step 8: Error Boundaries - Margin distribution, confidence
Step 9: Architecture Effects - Capacity, purity by architecture

Quick Start
------------
# Step 1: Voronoi cell visualization (2D)
python step1_voronoi_cells.py --dim 2 --epochs 50

# Step 2: Centroidal condition
python step2_centroidal.py --dim 2 --epochs 50

# Step 3: Energy minimization
python step3_energy.py --dim 2 --epochs 50

# Step 4: Flow dynamics
python step4_flow.py --dim 2 --epochs 50 --tau 0.1

# Step 5: Fixed-point analysis
python step5_fixedpoint.py --dim 8 --epochs 100

# Step 6: Collision sweep
python step6_collisions.py --dims 2,4,6,8,10,12,16 --epochs 80

# Step 7: Neural collapse
python step7_neural_collapse.py --dims 2,4,6,8,10 --epochs 100

# Step 8: Error boundaries
python step8_boundaries.py --dim 2 --epochs 50

# Step 9: Architecture comparison
python step9_architecture.py --dim 2 --epochs 50
python step9_architecture.py --dim 8 --epochs 50

Common Arguments
----------------
--dim      Feature dimension (2, 3, 4, 8)           Default: 2
--epochs   Number of training epochs               Default: 50
--full     Use full CIFAR-10 dataset (50k samples) Default: False
--quick    Quick mode (fewer epochs/dimensions)    Default: False

Step-Specific Arguments
-----------------------
Step 4: --tau      Temperature parameter (0.01, 0.1, 1.0)
Step 6: --dims     Comma-separated list of dimensions
Step 7: --dims     Comma-separated list of dimensions
Step 9: --dim      Feature dimension (2 or 8)

Output Directories
------------------
step1_results_cifar10_m2/      - Voronoi cell visualizations
step2_results_centroidal_m2/   - Centroidal error heatmaps
step3_results_energy_m2/       - Energy curves
step4_results_flow_m2_tau0.1/  - Alignment scatter plots
step5_results_fixedpoint_m8/   - Convergence comparisons
step6_results_collisions_sweep/ - Collision vs dimension plots
step7_results_neural_collapse/ - Regime comparison charts
step8_results_boundaries_m2/   - Margin distributions
step9_results_architecture_m2/ - Architecture comparisons

Expected Key Results
--------------------
Step 1: Cell purity 27% -> 72%
Step 2: Centroidal error 4.55 -> 0.0087
Step 3: Voronoi energy 7,601 -> 137,467
Step 4: Cosine similarity positive (+0.33)
Step 5: Stationarity ||∇_Z L|| = 0.0024
Step 6: Critical dimension m = 4 for collision < 10%
Step 7: Both regimes critical dimension m = 4
Step 8: p < 0.001 for margin difference
Step 9: Wide network best: 84.1% accuracy at m=8

Notes
-----
- First run automatically downloads CIFAR-10 dataset (~163 MB)
- Results saved as PNG images and PT/JSON data files
- Fixed random seed (42) for reproducibility
- GPU auto-detected; falls back to CPU

