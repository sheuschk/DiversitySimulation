import argparse
import sys
import os
import json
import numpy as np

from ..environments import create_env, NK_landscape_loaded
from datetime import datetime

from scipy.special import softmax

import pickle

from .helpers import run_intervention, loss_gradients, clip_steps, draw_step, bit_switch, parse_NK


def local_search_2D(env: np.array, agent_value, eta: float, position: np.array, steps, grid_size, tau, rng):
    """Stochastic Hill climbing OR random step (only if better)"""
    if eta >= rng.random():
        rand_pos = rng.integers(0, grid_size, 2)
        rand_value = env[rand_pos[0]][rand_pos[1]]
        if rand_value > agent_value:
            return rand_pos.reshape(1, 2), np.array([1.]), False
    neighbour_states = [[clip_steps(position[0]+s[0], grid_size), clip_steps(position[1]+s[1], grid_size)] for s in steps]
    ind_values = np.array([env[p[0]][p[1]] for p in neighbour_states])
    ind_values_pb = softmax((ind_values / tau))
    return np.asarray(neighbour_states), ind_values_pb


def local_search_NK(env, state, eta):        
    digits = rng.choice(20, size=(5), replace=False)
    neigh_states = np.zeros((len(digits) +1, env.N)).astype(int)
    neigh_states[0] = state 
    neigh_states[1:] = np.array([bit_switch(state.copy(), digit) for digit in digits])
    neigh_scores = np.array([env.get_fitness(ns) for ns in neigh_states])
    neigh_pb = softmax((neigh_scores * eta))
    return neigh_states, neigh_pb


def local_search(env_type, env, S, eta, solution, all_steps, grid_size, tau, rng):
    if env_type == "NK":
        new_solutions, probability_distribution = local_search_NK(env, solution, eta)
    elif env_type == "2D":
        new_solutions, probability_distribution = local_search_2D(env, S, eta, solution, all_steps, grid_size, tau, rng)
    return new_solutions, probability_distribution


def create_roleModel_features(n_roleModels, rng=None):
    """Feature matrix F (n_roleModels x features): F[agent_i] = [gamma, eta, n0, n1, ...]"""
    gammas = np.clip(rng.normal(0.8, 0.05, size=n_roleModels), 0, 1)
    etas = np.clip(rng.normal(0.8, 0.05, size=n_roleModels), 0, 1)
    noise = rng.random(n_roleModels)
    rho = rng.random(n_roleModels) * 0.2
    F = np.stack((gammas, etas, noise, rho), axis=1)
    return F


def initialize_role_models(env_type, env, n_roleModels, rng):
    """Init less privileged agents, but explicitly good performance and traits"""
    F = create_roleModel_features(n_roleModels)
    
    if env_type == "NK":
        fitness_values = list(env.fitness_dict.values())
        fitness_values = (fitness_values - min(fitness_values)) 
        fitness_values /= max(fitness_values)
        max_values = np.sort(fitness_values)[-100:]
        S = rng.choice(max_values, size=n_roleModels)
        # quantile = np.quantile(fitness_values, 0.99)
        # high_pos = np.where((np.round(quantile, 3) < np.round(fitness_values, 3)))[0]
        # indexes = rng.choice(high_pos, size=n_roleModels)
        # S = fitness_values[indexes]
        # states = np.array([env.get_fitness(np.array(list(f'{i:020b}')).astype(int)) for i in index])
        return F, S
        
    quantile = np.quantile(env, 0.9)
    high_pos = np.where((np.round(quantile, 3) < np.round(env, 3)))
    start_pos_index = rng.integers(0, len(high_pos[0]), n_roleModels)
    all_positions = np.stack((high_pos[0][start_pos_index], high_pos[1][start_pos_index]), axis=1)
    S = env[all_positions[:, 0], all_positions[:, 1]]
    return F, S

def initialize_starting_position_NK(env, n_agents, rho):
    fitness_values = list(env.fitness_dict.values())
    fitness_values = (fitness_values - min(fitness_values)) 
    fitness_values /= max(fitness_values)
    
    quantile = np.quantile(fitness_values, rho)
    minority_index = rho <= 0.2
    
    S = np.zeros(n_agents)
    states = np.empty((n_agents, N)).astype(int)
    
    for _, q in enumerate(quantile):
        if minority_index[_] == True:
            positions = np.where((np.round(q, 2) >= np.round(fitness_values, 2)))[0]
        else:
            positions = np.where((np.round(q, 2) <= np.round(fitness_values, 2)))[0]
            
        index = rng.choice(positions).astype(int)
        S[_] = fitness_values[index]
        states[_] = np.array(list(f'{index:020b}')).astype(int)
    return states, S

def initialize_starting_position_2D(env, n_agents, rho):
    minority_index = rho <= 0.2
    quantile = np.quantile(env, rho)

    solutions = np.empty((n_agents, 2), dtype=int)

    for _, q in enumerate(quantile):
        if minority_index[_] == True:
            q_solutions = np.where((np.round(q, 2) >= np.round(env, 2)))
            index = rng.integers(0, len(q_solutions[0]))
            solutions[_] = [q_solutions[0][index], q_solutions[1][index]]
        else:
            q_solutions = np.where((np.round(q, 2) <= np.round(env, 2)))
            index = rng.integers(0, len(q_solutions[0]))
            solutions[_] = [q_solutions[0][index], q_solutions[1][index]]
    
    S = env[solutions[:, 0], solutions[:, 1]]
    return solutions, S

def initialize_starting_position(env_type, env, n_agents, rho):
    if env_type == "NK":
        return initialize_starting_position_NK(env, n_agents, rho)
    elif env_type == "2D":
        return initialize_starting_position_2D(env, n_agents, rho)


def run_simulation(args, F, env, rng):
    """Run simulation"""
    env_type = args["env_type"]
    n_agents = args["n_agents"]
    intervention = args["intervention"]
    all_steps = np.array([[i, j] for i in np.arange(-1, 2) for j in np.arange(-1, 2)])

    gammas, etas, rho = F[:,0] * 0.1, F[:,1]*0.1, F[:,-1]

    all_solutions, S = initialize_starting_position(env_type, env, n_agents, rho)


    # initialize
    F_rm = R_rm = None
    if intervention == True:
        F_rm, R_rm = initialize_role_models(env_type, env, args["n_roleModels"], rng)
    
    W = rng.normal((max(S)-min(S))*0.25, 0.05, (n_agents, 4))
    A = W @ F.T


    # Track simulation
    intermediate_solutions = np.zeros_like(all_solutions)
    intermediate_S = np.zeros_like(S)
    S_history = np.empty((args["max_iter"], n_agents))
    S_history[0] = S
    R = S

    # Repeated processes
    for t in range(args["max_iter"]):
        for agent_i in range(n_agents):
            # (1) & (2) Individual and social optimization
            if gammas[agent_i] >= rng.random():
                probability_distribution = softmax((A/A.sum(axis=1)) / args["tau"], axis=1)
                new_solutions = all_solutions
            else:
                new_solutions, probability_distribution = local_search(env_type, env, S[agent_i], etas[agent_i],
                                                             all_solutions[agent_i], all_steps,
                                                             args["grid_size"], args["tau"], rng)
            next_solution = draw_step(new_solutions, probability_distribution)
            intermediate_solutions[agent_i] = next_solution
            intermediate_S[agent_i] = env.get_fitness(next_solution) if env_type == "NK" else env[next_solution[0], next_solution[1]]

        # (3) Solution implementation
        S = intermediate_S
        all_solutions = intermediate_solutions
        S_history[t] = S
        R_decay = R_scale[-t:] / R_scale[-t:].sum()
        R = R_decay @ S_history[:t]

        # (4) Social influence matrix update
        b = R.min()
        R_hat = W @ F.T + b
        grad = loss_gradients(R_hat, F, R)
        W -= (args["lr"] * grad.T).T

        if intervention is True:
            if (50 < t <= 50 + args["int_duration"]):
                grad_rm = run_intervention(args["n_roleModels"], args["sigma"], F, W, F_rm, R_rm, b, rng)
                W -= args["lr"] * grad_rm


    return S_history, W, A, W_history



def init_parser():
    """ Parse arguments for the script"""
    parser = argparse.ArgumentParser(description='Run Simulation')

    parser.add_argument("--env", '-e', help="Fitness environment", default="MasonWatts")
    parser.add_argument("--env_type", '-et', help="NK or 2D", default="NK", choices=["NK", "2D"])

    parser.add_argument("--intervention", '-in', help="intervention", default=False, type=bool)
    parser.add_argument("--int_duration", '-id', help="Duration of intervention", default=10, type=int)

    parser.add_argument("--total_simulations", '-ts', help="Total number of simulations", default=1000, type=int)

    parser.add_argument("--max_iter", '-i', help="Number of Iterations", default=250, type=int)
    parser.add_argument("--n_agents", '-n', help="Number of agents", default=7, type=int)

    parser.add_argument("--seed", '-s', help="Seed", default=42, type=int)

    args = vars(parser.parse_args())

    # Add aditional parameters
    args["grid_size"] = 1000
    args["lambda_scale"]       = 0.9  # Label decay
    args["beta"]               = 0.5  # Learning rate for social influence matrix A
    args["tau"]                = 0.01  # Exploitation for social and individual optimization
    args["sigma"]              = 0.2  # Probability for mentor connections during continuous intervention
    args["intervention_start"] = 50
    args["lr"]                 = 0.1  # Learning rate for individual optimization

    all_arg_keys = set(["lr", "tau", "beta", "lambda_scale", "sigma", "intervention_start", "int_duration",
                     "n_agents", "max_iter", "env", "env_type", "seed", "intervention", "total_simulations", "grid_size"])
    assert len(set(args.keys()).intersection(all_arg_keys)) == len(all_arg_keys), f"Not all necessary arguments are given only: {set(args.keys()).intersection(all_arg_keys)}"


    sys.stdout.write(f"Param: value \n")
    for arg in args.keys():
        sys.stdout.write(f"{arg}: {args[arg]} \n")
    return args

def save_results(args, file_title):
    if not os.path.exists("results"):
        os.mkdir("results")

    inter = 0
    if args["intervention"] is True:
        inter = f"{args['int_duration']}"
    file_title += inter
    dir_name = f"results/arg_{args['env']}"

    time = datetime.now().strftime('%Y_%m_%dT%H_%M_%S')
    dir_name += f"/{time}"
    os.mkdir(dir_name)
    return dir_name, file_title

if __name__ == '__main__':
    """Simulation 1: Individual optimization in a collective with privileg and mentorship"""
    sys.stdout.write('\n Script is starting\n')
    args = init_parser()

    # Simulation parameter
    total_simulations          = args["total_simulations"]
    n_agents                   = args["n_agents"]
    max_iterations             = args["max_iter"]
    R_scale = args["lambda_scale"] ** np.arange(max_iterations - 1, -1, -1)

    # Create environment
    sys.stdout.write('\nStarting to read environment\n')
    if args["env_type"] == "NK":
        N, K, n_landscapes = parse_NK(args['env'])
        file_title = f"N{N}K{K}"
        with open(args["env"], 'rb') as f:
            landscapes = pickle.load(f)
        for i in range(n_landscapes):
            keys = landscapes[i].keys()
            values = np.array(list(landscapes[i].values()))
            values = values ** 4
            landscapes[i] = dict(zip(keys, values))
    elif args["env_type"] == "2D":
        env, file_title = create_env(args["env"], args["grid_size"])
    sys.stdout.write('\nFinished with reading environment\n')

    # Result saving
    all_results = np.empty((total_simulations*n_agents, 13))  # Second param: How many vars to track
    W_results = np.zeros((max_iterations, 4))
    dir_name, file_title = save_results(args, file_title)
    np.save(f"{dir_name}/{file_title}", np.array(all_results))
    with open(f"{dir_name}/parameter.json", "w") as outfile:
        json.dump(args, outfile)

    # Start simulations
    base_rng = np.random.default_rng(seed=args["seed"])
    sys.stdout.write(f"\nIn total {total_simulations} simulations")
    for _ in range(total_simulations):
        # Create traits
        rng = np.random.default_rng(seed=base_rng.integers(0, 100000))

        if args["env_type"] == "NK":
            env = NK_landscape_loaded(N, K, landscapes[rng.integers(0, n_landscapes)])

        F = rng.random((n_agents, 4))
        S_history, W, A, W_history = run_simulation(args, F, env, rng)

        run_results = np.concatenate((F,
                                      W,
                                      S_history.max(axis=0).reshape(n_agents, 1),
                                      S_history.mean(axis=0).reshape(n_agents, 1),
                                      S_history[0].reshape(n_agents, 1),
                                      A.mean(axis=0).reshape(n_agents, 1),
                                      ), axis=1)

        all_results[_*n_agents: _*n_agents+n_agents] = run_results

        W_results += W_history

        if _ % 20 == 0:
            sys.stdout.write(f"\n Simulation: {_}")
        if _ % 50 == 0:
            np.save(f"{dir_name}/{file_title}", np.array(all_results))
    W_results /= total_simulations

    np.save(f"{dir_name}/{file_title}_weights", np.array(W_results))
    np.save(f"{dir_name}/{file_title}", np.array(all_results))
    sys.stdout.write(f"\nFinished successfully")
