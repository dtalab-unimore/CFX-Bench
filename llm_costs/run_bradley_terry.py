import argparse
import json
import os

import pandas as pd

import numpy as np

from llm_costs.utils import setup_logging


def add_pseudocounts(wins, alpha=0.5):
    """
    Add a small symmetric pseudocount to every off-diagonal cell of the wins
    matrix. This guarantees Hunter's Assumption 1 holds (every feature has at
    least one 'win' and one 'loss' against every other feature), so the
    Bradley-Terry MLE is finite and unique.

    Parameters
    ----------
    wins : (m, m) int array
        Raw wins matrix from elicitation.
    alpha : float
        Pseudocount per ordered pair. alpha=0.5 means we pretend one extra
        comparison per unordered pair, split half-half. Hunter (2004) suggests
        "a tiny fraction of a win" for this purpose.

    Returns
    -------
    smoothed : (m, m) float array
        wins with alpha added to every off-diagonal cell.
    """
    smoothed = wins.astype(float).copy()
    m = smoothed.shape[0]
    off_diag = ~np.eye(m, dtype=bool)
    smoothed[off_diag] += alpha
    return smoothed


def fit_bradley_terry_mm(wins, tol=1e-9, max_iter=10_000):
    """
    Fit a Bradley-Terry model via Hunter's MM algorithm.

    Parameters
    ----------
    wins : (m, m) array of nonnegative ints/floats
        wins[i, j] = number of times feature i was judged HARDER
        to change than feature j.
    tol : float
        L2 convergence tolerance on the gamma vector.
    max_iter : int
        Safety cap on iterations.

    Returns
    -------
    gamma : (m,) array, normalized so sum(gamma) = 1
        Multiplicative strengths.
    beta : (m,) array
        Log-strengths (= log gamma), anchored so beta[0] = 0.
        These are the feature costs.
    n_iter : int
        Iterations used.
    """
    wins = np.asarray(wins, dtype=float)
    m = wins.shape[0]
    assert wins.shape == (m, m), "wins must be square"
    assert np.all(np.diag(wins) == 0), "diagonal must be zero"

    # N[i, j] = total number of i-vs-j comparisons (symmetric)
    N = wins + wins.T

    # W[i] = total wins for feature i
    W = wins.sum(axis=1)

    # Initialize uniformly
    gamma = np.ones(m) / m

    for k in range(1, max_iter + 1):
        gamma_new = np.empty(m)
        for i in range(m):
            denom = 0.0
            for j in range(m):
                if i == j:
                    continue
                denom += N[i, j] / (gamma[i] + gamma[j])
            # Hunter (2004), eq. (3)
            gamma_new[i] = W[i] / denom if denom > 0 else gamma[i]

        # Renormalize to the simplex (identifiability constraint)
        gamma_new /= gamma_new.sum()

        if np.linalg.norm(gamma_new - gamma) < tol:
            gamma = gamma_new
            break
        gamma = gamma_new

    # Convert to log scale, anchored at feature 0 (beta[0] = 0)
    beta = np.log(gamma) - np.log(gamma[0])
    return gamma, beta, k


def pretty_print_costs(feature_names, beta, gamma, save_path=None):
    """Print feature costs sorted from hardest to easiest to change."""
    order = np.argsort(-beta)
    costs = beta - beta.min() + 1.0
    header = f"{'feature':<20}{'beta (cost)':>14}{'gamma':>12}{'add cost':>12}"
    print(header)
    print("-" * len(header))
    for idx in order:
        print(f"{feature_names[idx]:<20}{beta[idx]:>14.3f}{gamma[idx]:>12.4f}{costs[idx]:>12.3f}")

    if save_path is not None:
        data = np.stack([beta, gamma, costs], axis=1)
        data = pd.DataFrame(data, columns=['beta', 'gamma', 'costs'], index=feature_names)
        data.to_csv(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run Bradley-Terry with LLM.',
        usage='run_bradley_terry.py [<args>], [-h | --help]'
    )
    parser.add_argument('--dataset', type=str, required=True, choices=[
        'german-credit', 'german-credit-crif', 'german-credit-crif-full',
        'lending-club', 'lending-club-2', 'lending-club-3'
    ])
    parser.add_argument('--model', type=str, required=True, choices=[
        'llama-3.1-8b', 'qwen', 'gpt-4o-mini', 'mistral', 'gpt-5-mini', 'gpt-5.4-mini'
    ])
    parser.add_argument('--n-repeats', type=int, default=5)
    parser.add_argument('--ordering', type=str, choices=["shuffle", "i_first", "j_first"], default="shuffle")
    parser.add_argument('--seed', type=int, default=0)
    args = vars(parser.parse_args())

    dataset = args['dataset']

    output_dir = os.path.join(
        "output", "llm_costs", dataset.replace('-', '_'),
        f"{args['model']}__n{args['n_repeats']}__{args['ordering']}__s{args['seed']}"  # experiment tag
    )
    output_dir += "__prompt=inc_cat_v2"

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"{output_dir} does not exist")
    logger = setup_logging(output_dir + "/bt.log")

    # Load wins
    wins = pd.read_csv(f"{output_dir}/wins_matrix.csv", index_col=0)
    feature_keys = wins.index
    wins = wins.to_numpy()

    # Smooth the wins matrix so every feature has nonzero wins and losses.
    # This prevents BT from diverging on features that lost (or won) every comparison.
    wins = add_pseudocounts(wins, alpha=0.5)
    logger.info("Applied pseudocounts (alpha=0.5) before BT fitting")

    # Plug into Bradley-Terry:
    gamma, beta, n_iter = fit_bradley_terry_mm(wins)
    logger.info("BT converged in %d iterations", n_iter)
    pretty_print_costs(feature_keys, beta, gamma, save_path=(output_dir+'/bt_costs.csv'))
