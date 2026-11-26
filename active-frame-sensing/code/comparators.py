"""Additional controls for the decision target and the graph aggregation architecture."""
import csv

import numpy as np
import torch
from torch import nn

from decision import Tree, tied_min
from evaluate import write
from make_tasks import ROOT


def branch_entropy(action_mass):
    """Branch probability times the conditional entropy of a sampled optimal command."""
    total = action_mass.sum(axis=-1, keepdims=True)
    normalized = np.divide(action_mass, total, out=np.zeros_like(action_mass), where=total > 0)
    logprob = np.log(np.where(normalized > 0, normalized, 1))
    return -(action_mass * logprob).sum(axis=-1)


def action_entropy_cost(tree):
    # A latent reference action is chosen uniformly from the shortest commands for its frame.
    optimal = (tree.c == 0).astype(float)
    optimal /= optimal.sum(axis=1, keepdims=True)
    first = tied_min(branch_entropy(tree.one @ optimal).sum(axis=1))
    second_entropy = branch_entropy(tree.joint @ optimal).sum(axis=-1)
    second = tied_min(np.where(tree.valid, second_entropy, np.inf))
    return tree.score(first, second)[0]


def fit_dense(features, targets, fit_ids, val_ids, seed):
    torch.manual_seed(seed)
    random = np.random.default_rng(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(288, 14), nn.ReLU(), nn.Linear(14, 12))
    assert sum(p.numel() for p in model.parameters()) == 4226
    optimizer = torch.optim.Adam(model.parameters(), lr=.0025)
    best_loss = float('inf')
    for epoch in range(1, 81):
        for batch in np.array_split(random.permutation(fit_ids), 16):
            loss = (model(features[batch]) - targets[batch]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            loss = (model(features[val_ids]) - targets[val_ids]).square().mean().item()
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    torch.save(best, ROOT / 'data' / f'dense_probe_{seed}.pt')
    with torch.no_grad():
        return model(features).numpy(), best_epoch


def main():
    torch.set_num_threads(2)
    with (ROOT / 'data/tasks.csv').open() as f:
        tasks = list(csv.DictReader(f))
    data = np.load(ROOT / 'data/tasks.npz')
    supervision = np.load(ROOT / 'data/graph_training.npz')
    features = torch.from_numpy(supervision['features'])
    targets = torch.from_numpy(supervision['targets'])
    splits = np.array([r['split'] for r in tasks])
    heldout = np.flatnonzero(np.isin(splits, ['test', 'closure', 'city']))
    seeds = (83, 113, 191)
    fits = {seed: fit_dense(features, targets, np.flatnonzero(splits == 'fit'),
                            np.flatnonzero(splits == 'validation'), seed) for seed in seeds}
    rows = []
    for i in heldout:
        tree = Tree(data['signature'][i], data['cost'][i])
        baseline = action_entropy_cost(tree)
        values = [tree.score(tied_min(fits[seed][0][i]), tree.second)[0] for seed in seeds]
        optimum = float(tree.q.min())
        assert min([baseline, *values]) >= optimum - 1e-10
        rows.append((i, splits[i], tasks[i]['source'], baseline, *values, np.mean(values), optimum))
    write('comparison_controls.csv',
          ['index','split','source','action_entropy','dense_83','dense_113','dense_191','dense_mean','adaptive'], rows)
    print('Validation-selected dense epochs:', [fits[s][1] for s in seeds])
    for split in ['test', 'closure', 'city']:
        scores = np.array([r[3:] for r in rows if r[1] == split])
        print(split, 'action entropy, dense, optimum', scores.mean(axis=0)[[0, -2, -1]])


if __name__ == '__main__':
    main()

