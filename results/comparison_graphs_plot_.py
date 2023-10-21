import json

import matplotlib.pyplot as plt
import torch
import numpy as np
from topologies import Matrix
from demo import simulate_random_walk
import seaborn as sns
import pandas as pd
from STAT_TEST import TimeVaringErdos, Erdos
from tqdm import trange, tqdm
from OurTopologies import Star, Ring, TwoCliques, CycleChangingGraph, SlowChangingRing
from random_walks import effective_number_of_neighbors
def get_best_results(dict_logs):
    new_dict = {}
    for key in dict_logs:
        new_dict[key] = min([dict_logs[key][value] for value in dict_logs[key] if not np.isnan(dict_logs[key][value])])
    return new_dict
def spectral_gap(matrix):
    _, s, _ = torch.linalg.svd(matrix)
    abs_eigenvalues = torch.sqrt(s**2)
    return abs_eigenvalues[0] - abs_eigenvalues[1]
#
#
with open('new_experiment_now_ok.json', 'r') as f:
    ful_exp = json.load(f)
#
# with open('second_exp.json', 'r') as f:
#     second_exp = json.load(f)
#
with open('new_experiment_now_ok_last_three.json', 'r') as f:
    ful_exp_three = json.load(f)
#
# print(ful_exp)
# print(third_exp)
#
# print(get_best_results(ful_exp))
# print(get_best_results(third_exp))
#
ful_exp = get_best_results(ful_exp)
ful_exp.update(get_best_results(ful_exp_three))
#
print(ful_exp)
num_workers = 16
device = 'cpu'
# #
# #
W_fully = Matrix((torch.ones((num_workers, num_workers)) / num_workers).to(device))
W_eye = Matrix(torch.eye(num_workers).to(device))
scheme_var = TimeVaringErdos(n=num_workers, p=1/4)
scheme_const = Erdos(n=num_workers, p=1/4)
#
star = Star(num_workers - 1)
two_cliques = TwoCliques(num_workers)
ring = Ring(num_workers)
#
star_ = Star(num_workers - 1)
two_cliques_ = TwoCliques(num_workers)
ring_ = Ring(num_workers)
cycle_changing = CycleChangingGraph(num_workers, [star_, two_cliques_, ring_])

Erdos_p = [Erdos(num_workers, p) for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
Erdos_names = [f'Erdos_{p}' for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

Var_Erdos_p = [TimeVaringErdos(num_workers, p) for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
Var_Erdos_names = [f'Var_Erdos_{p}' for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

changing_ring = SlowChangingRing(num_workers, 2)

# schemes = [W_fully, W_eye, scheme_var, scheme_const, star, two_cliques, ring, cycle_changing, changing_ring] + Erdos_p + Var_Erdos_p
# names = ['fully', 'solo', 'Erdos_1/4_var', 'Erdos_1/4_const', 'star', 'two_cliques', 'ring', 'cycle_changing', 'changing_ring'] + Erdos_names + Var_Erdos_names

schemes = [W_fully, W_eye, scheme_var, scheme_const, star, two_cliques, ring, cycle_changing, changing_ring]
names = ['fully', 'solo', 'Erdos_1/4_var', 'Erdos_1/4_const', 'star', 'two_cliques', 'ring', 'cycle_changing', 'changing_ring']

#

names_rights = []


print(len(schemes))
dict_with_enn = {}
dict_with_lambdas = {}
#

enns = []
lambdas = []
results = []

for name, scheme in tqdm(zip(names, schemes)):
    if name in ful_exp:
        names_rights.append(name)

        results.append(ful_exp[name])

        samples = []
        eff_num_neigh = simulate_random_walk(scheme, gamma=0.951, num_steps=1000, num_workers=16, num_reps=1000)

        dict_with_enn[name] = eff_num_neigh
        enns.append(eff_num_neigh)
        G_s = []
        for step in range(2000):
            new_G = scheme.w(step)
            try:
                G_s.append(spectral_gap(new_G))
            except:
                pass
        lambdas.append(np.mean(G_s))
        dict_with_lambdas[name] = np.mean(G_s)

stacked_np = np.zeros((len(names_rights), 3))
stacked_np[:, 0] = np.array(results)
stacked_np[:, 1] = np.array(enns)
stacked_np[:, 2] = np.array(lambdas)
# stacked_np = np.load('results_new.npy')

stacked_df = pd.DataFrame(stacked_np, columns=['opt_rates', 'enns', 'spectral_gaps'])

fig, (ax1, ax2) = plt.subplots(1, 2)

fig.set_figheight(8)
fig.set_figwidth(15)

p1 = sns.scatterplot(ax=ax1, data=stacked_df, x='enns', y='opt_rates')

names = names_rights

for idx in range(stacked_np.shape[0]):
     p1.text(stacked_df.enns[idx]+0.01, stacked_df.opt_rates[idx], names_rights[idx],
     horizontalalignment='left',
     size='small', color='black')

ax1.set_ylabel('Final loss')
ax1.set_xlabel('Effective number of neighbours')
ax1.grid()

p2 = sns.scatterplot(ax=ax2, data=stacked_df, x='spectral_gaps', y='opt_rates')

for idx in range(stacked_np.shape[0]):
     p2.text(stacked_df.spectral_gaps[idx]+0.01, stacked_df.opt_rates[idx], names_rights[idx],
     horizontalalignment='left',
     size='small', color='black')

ax2.set_ylabel('Final loss')
ax2.set_xlabel('Spectral gap')
ax2.grid()
plt.savefig('right_results_without_erdos')

np.save('results_new_without_erdos', stacked_np)
