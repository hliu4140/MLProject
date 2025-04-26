import json
import numpy as np
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
from scipy.optimize import dual_annealing

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    less = np.array(data['less_toxic'])
    more = np.array(data['more_toxic'])
    return less, more

def compute_loss(wt, less, more):
    less_score = less @ wt
    more_score = more @ wt
    return 1 - np.mean(less_score < more_score)

def spsa(less, more, wt0=None, N=5000, a=0.5, A=50, c=0.1,
         alpha=0.602, gamma=0.101, perturbation='bernoulli', seed=None):
    p = less.shape[1]
    if wt0 is None:
        wt = np.ones(p) / p
    else:
        wt = wt0.copy()
    if seed is not None:
        np.random.seed(seed)
    best_wt = wt.copy()
    min_loss = compute_loss(wt, less, more)
    trace = []
    for k in range(1, N+1):
        a_k = a / ((k + 1 + A) ** alpha)
        c_k = c / ((k + 1) ** gamma)
        if perturbation == 'bernoulli':
            delta = np.random.choice([-1, 1], size=p)
        elif perturbation == 'uniform':
            delta = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=p)
        elif perturbation == 'normal':
            delta = np.random.normal(0, 1, size=p)
        else:
            raise ValueError("Unknown perturbation type.")
        y_plus = compute_loss(wt + c_k * delta, less, more)
        y_minus = compute_loss(wt - c_k * delta, less, more)
        ghat = (y_plus - y_minus) / (2 * c_k * delta)
        wt = wt - a_k * ghat
        wt = np.clip(wt, 0, 1)
        if wt.sum() != 0:
            wt = wt / wt.sum()
        else:
            wt = np.ones_like(wt) / len(wt)
        current_loss = compute_loss(wt, less, more)
        if current_loss < min_loss:
            min_loss = current_loss
            best_wt = wt.copy()
        trace.append(current_loss)
    return trace, best_wt, min_loss

def run_genetic_algorithm(less, more, seed=None,
                          max_iter=200, pop_size=100, mutation_prob=0.1,
                          elit_ratio=0.05, crossover_prob=0.5, parents_portion=0.3,
                          crossover_type='uniform', max_no_improve=20):
    p = less.shape[1]
    loss_trace = []
    def loss_wrapper_ga(wt):
        wt = np.clip(wt, 0, 1)
        if wt.sum() > 0:
            wt = wt / wt.sum()
        else:
            wt = np.ones_like(wt) / len(wt)
        val = compute_loss(wt, less, more)
        loss_trace.append(val)
        return val
    if seed is not None:
        np.random.seed(seed)
    varbound = np.array([[0, 1]] * p)
    model = ga(function=loss_wrapper_ga,
               dimension=p,
               variable_type='real',
               variable_boundaries=varbound,
               algorithm_parameters={
                   'max_num_iteration': max_iter,
                   'population_size': pop_size,
                   'mutation_probability': mutation_prob,
                   'elit_ratio': elit_ratio,
                   'crossover_probability': crossover_prob,
                   'parents_portion': parents_portion,
                   'crossover_type': crossover_type,
                   'max_iteration_without_improv': max_no_improve
               })
    model.run()
    best_wt = np.array(model.output_dict['variable'])
    best_wt = best_wt / best_wt.sum() if best_wt.sum() != 0 else np.ones_like(best_wt) / len(best_wt)
    best_loss = model.output_dict['function']
    return loss_trace, best_wt, best_loss

def run_simulated_annealing(less, more, seed=None):
    p = less.shape[1]
    loss_trace = []
    def loss_wrapper_sa(wt):
        wt = np.clip(wt, 0, 1)
        if wt.sum() > 0:
            wt = wt / wt.sum()
        else:
            wt = np.ones_like(wt) / len(wt)
        val = compute_loss(wt, less, more)
        loss_trace.append(val)
        return val
    sa_trace = []
    def callback_sa(x, f, context):
        sa_trace.append(f if not sa_trace else min(sa_trace[-1], f))
    bounds = [(0, 1)] * p
    if seed is not None:
        np.random.seed(seed)
    result = dual_annealing(loss_wrapper_sa, bounds, callback=callback_sa)
    best_wt = result.x
    best_loss = result.fun
    return sa_trace, best_wt, best_loss

def plot_loss_curves(curves):
    plt.figure(figsize=(8, 5))
    for label, trace in curves.items():
        plt.plot(trace, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    less, more = load_data('distilbert_logits.json')
    # Run SPSA
    spsa_trace, spsa_wt, spsa_loss = spsa(less, more, N=5000, seed=66)
    print(f"SPSA best weight: {spsa_wt}, loss: {spsa_loss:.4f}")
    # Run Genetic Algorithm
    ga_trace, ga_wt, ga_loss = run_genetic_algorithm(less, more, seed=66)
    print(f"GA best weight: {ga_wt}, loss: {ga_loss:.4f}")
    # Run Simulated Annealing
    sa_trace, sa_wt, sa_loss = run_simulated_annealing(less, more, seed=66)
    print(f"SA best weight: {sa_wt}, loss: {sa_loss:.4f}")
    # Plot comparison
    plot_loss_curves({'SPSA': spsa_trace, 'GA': ga_trace, 'SA': sa_trace})

if __name__ == '__main__':
    main()