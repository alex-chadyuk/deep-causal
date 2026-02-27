"""
Structural Causal Model (SCM) data generator.

Generates a random DAG over variables, defines structural equations with random
coefficients and additive noise, and produces synthetic data plus graph visualization.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


def build_random_dag(
    n_variables: int,
    edge_prob: float = 0.4,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """
    Build a random directed acyclic graph (DAG) over n_variables nodes.

    Nodes are in topological order 0, 1, ..., n_variables-1. An edge i -> j
    exists only when i < j, with probability edge_prob per pair.

    Parameters
    ----------
    n_variables : int
        Number of variables (nodes).
    edge_prob : float, optional
        Probability of each possible edge (i, j) with i < j. Default 0.4.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        DAG with nodes 0..n_variables-1 and edges (i, j) for i < j.
    """
    if seed is not None:
        random.seed(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_variables))
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            if random.random() < edge_prob:
                G.add_edge(i, j)
    return G


def build_structural_equations(
    G: nx.DiGraph,
    noise_std: float = 1.0,
    coef_low: float = -1.0,
    coef_high: float = 1.0,
    intercept_low: float = -0.5,
    intercept_high: float = 0.5,
    seed: Optional[int] = None,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, float], dict[int, float]]:
    """
    Define structural equations for each node: X_j = intercept_j + sum_i coef_ij * X_i + N(0, noise_std_j).

    Returns
    -------
    parent_coefs : dict
        parent_coefs[j] = [(i, coef_ij), ...] for each parent i of j.
    intercepts : dict
        intercepts[j] = intercept for node j.
    noise_stds : dict
        noise_stds[j] = standard deviation of additive Gaussian noise for node j.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    parent_coefs: dict[int, list[tuple[int, float]]] = {j: [] for j in G.nodes()}
    intercepts: dict[int, float] = {}
    noise_stds: dict[int, float] = {}

    for j in G.nodes():
        parents = list(G.predecessors(j))
        intercepts[j] = float(
            rng.uniform(intercept_low, intercept_high)
        )
        noise_stds[j] = float(rng.uniform(0.2, noise_std))
        for i in parents:
            c = float(rng.uniform(coef_low, coef_high))
            parent_coefs[j].append((i, c))

    return parent_coefs, intercepts, noise_stds


def generate_data(
    G: nx.DiGraph,
    parent_coefs: dict[int, list[tuple[int, float]]],
    intercepts: dict[int, float],
    noise_stds: dict[int, float],
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate n_samples rows of data from the structural equations in topological order.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_variables). Column j corresponds to variable j.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    order = list(nx.topological_sort(G))
    n_vars = G.number_of_nodes()
    data = np.zeros((n_samples, n_vars))

    for j in order:
        intercept = intercepts[j]
        noise = rng.normal(0, noise_stds[j], size=n_samples)
        value = intercept + noise
        for i, coef in parent_coefs[j]:
            value = value + coef * data[:, i]
        data[:, j] = value

    return data


def show_graph(
    G: nx.DiGraph,
    variable_names: Optional[list[str]] = None,
    title: str = "Structural causal graph",
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """
    Draw the DAG using NetworkX and matplotlib.

    Parameters
    ----------
    G : nx.DiGraph
        The structural graph (nodes are int indices).
    variable_names : list of str, optional
        Labels for nodes. If None, uses "X0", "X1", ...
    title : str
        Plot title.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    figsize : tuple
        Figure size (width, height).
    """
    import matplotlib.pyplot as plt

    n = G.number_of_nodes()
    if variable_names is None:
        variable_names = [f"X{i}" for i in range(n)]
    elif len(variable_names) != n:
        variable_names = [f"X{i}" for i in range(n)]

    labels = {i: variable_names[i] for i in range(n)}

    pos = nx.spring_layout(G, seed=42, k=1.5)
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, ax=ax, arrowstyle="-|>")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def run_generator(
    n_variables: int = 5,
    n_samples: int = 1000,
    edge_prob: float = 0.4,
    noise_std: float = 1.0,
    variable_names: Optional[list[str]] = None,
    seed: Optional[int] = None,
    csv_path: Optional[str | Path] = None,
    show_plot: bool = True,
    plot_save_path: Optional[str | Path] = None,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """
    End-to-end: build DAG, structural equations, generate data, optionally plot and save.

    Parameters
    ----------
    n_variables : int
        Number of variables.
    n_samples : int
        Number of samples to generate.
    edge_prob : float
        Probability of each directed edge in the random DAG.
    noise_std : float
        Upper bound for per-node noise standard deviation.
    variable_names : list of str, optional
        Names for columns (and graph labels). Default "X0", "X1", ...
    seed : int, optional
        Random seed for DAG, coefficients, and data.
    csv_path : str or Path, optional
        Where to save the generated CSV.
    show_plot : bool
        Whether to call show_graph (opens matplotlib window).
    plot_save_path : str or Path, optional
        Where to save the graph figure.

    Returns
    -------
    G : nx.DiGraph
        The structural graph.
    df : pd.DataFrame
        Generated dataset with columns = variable names.
    """
    G = build_random_dag(n_variables, edge_prob=edge_prob, seed=seed)
    parent_coefs, intercepts, noise_stds = build_structural_equations(
        G, noise_std=noise_std, seed=(seed + 1) if seed is not None else None
    )
    data = generate_data(
        G, parent_coefs, intercepts, noise_stds, n_samples,
        seed=(seed + 2) if seed is not None else None,
    )

    if variable_names is None:
        variable_names = [f"X{i}" for i in range(n_variables)]
    df = pd.DataFrame(data, columns=variable_names)

    if show_plot:
        show_graph(
            G,
            variable_names=variable_names,
            save_path=plot_save_path,
        )

    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    return G, df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SCM data generator")
    parser.add_argument("-n", "--n-variables", type=int, default=5, help="Number of variables")
    parser.add_argument("-s", "--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("-p", "--edge-prob", type=float, default=0.4, help="Edge probability in DAG")
    parser.add_argument("--noise-std", type=float, default=1.0, help="Noise std scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--no-plot", action="store_true", help="Skip showing graph")
    parser.add_argument("--plot-output", type=str, default=None, help="Save graph figure to path")
    args = parser.parse_args()

    G, df = run_generator(
        n_variables=args.n_variables,
        n_samples=args.n_samples,
        edge_prob=args.edge_prob,
        noise_std=args.noise_std,
        seed=args.seed,
        csv_path=args.output,
        show_plot=not args.no_plot,
        plot_save_path=args.plot_output,
    )
    print(f"Generated DAG with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Dataset shape: {df.shape}")
    if args.output:
        print(f"Saved to {args.output}")
