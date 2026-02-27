# Data generator

Generates synthetic data from a random **Structural Causal Model (SCM)**: a directed acyclic graph (DAG) over variables with structural equations that include random coefficients and additive Gaussian noise.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Command line

```bash
# Default: 5 variables, 1000 samples, show graph
python scm_generator.py

# Custom size and output
python scm_generator.py -n 8 -s 2000 -o data.csv --seed 42

# Save graph figure and skip interactive plot
python scm_generator.py -n 6 -o data.csv --no-plot --plot-output graph.png
```

Options:

- `-n, --n-variables`  Number of variables (default: 5)
- `-s, --n-samples`    Number of samples (default: 1000)
- `-p, --edge-prob`    Edge probability in random DAG (default: 0.4)
- `--noise-std`        Scale for noise standard deviations (default: 1.0)
- `--seed`             Random seed
- `-o, --output`       Output CSV path
- `--no-plot`          Do not show the graph window
- `--plot-output`      Path to save the graph figure

### From Python

```python
from scm_generator import (
    build_random_dag,
    build_structural_equations,
    generate_data,
    show_graph,
    run_generator,
)

# One-shot: build DAG, equations, generate data, plot, save CSV
G, df = run_generator(
    n_variables=5,
    n_samples=1000,
    edge_prob=0.4,
    seed=42,
    csv_path="data.csv",
    show_plot=True,
    plot_save_path="graph.png",
)

# Or step by step
G = build_random_dag(5, edge_prob=0.4, seed=42)
parent_coefs, intercepts, noise_stds = build_structural_equations(G, seed=42)
data = generate_data(G, parent_coefs, intercepts, noise_stds, n_samples=1000, seed=42)
show_graph(G, title="My SCM")
```

Structural equation for each variable \(X_j\):

\[
X_j = \text{intercept}_j + \sum_{i \in \text{pa}(j)} \beta_{ij} X_i + \varepsilon_j, \quad \varepsilon_j \sim \mathcal{N}(0, \sigma_j^2)
\]

Coefficients and noise scales are drawn at random when building the model.
