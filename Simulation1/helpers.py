import numpy as np

def loss_fn(R_hat, R):
    """
    Mean Squared error
    R: Expected Rewards of all Agents
    S: Label
    """
    return 1/(len(R)) * np.sum(((R - R_hat) ** 2), axis=1)


def loss_gradients(R_hat, F, R):
    """
    Derivative of Mean Squared Error
    OMEGA: All Weights
    F: Feature matrix of all agents
    S: Label
    """
    dL = - 1 / (len(R)) * (R - (R_hat)) @ (2 * F)
    return dL

def clip_steps(position, grid_size):
    return np.clip(position, 0, grid_size - 1)

def draw_step(positions, prob_dist, rng):
    """@:return: (array [x1, x2]) the position"""
    return rng.choice(positions, p=prob_dist)

def bit_switch(state, digit):
    if state[digit] == 1:
        state[digit] = 0
    else:
        state[digit] = 1
    return state

def intervention_gradient(W, F_rm, R_rm, b, n_roleModels, chosen_agents, rng):
    """Calculate gradients for agents impacted by a mentor"""
    roleModels = rng.integers(n_roleModels, size=len(chosen_agents))
    R_hat_rm = W[chosen_agents] @ F_rm[roleModels].T + b
    W_grad = np.zeros_like(W)
    if len(R_hat_rm) > 0:
        grad_rm = loss_gradients(R_hat_rm, F_rm[roleModels], R_rm[roleModels])
        W_grad[chosen_agents] = grad_rm
    return W_grad

def run_intervention(n_roleModels, sigma, F, W, F_rm, R_rm, b, rng):
    """Run intervention which mentor impacts which less privileged agent"""
    chosen_agents = np.where(F[:, -1] < 0.2)[0]
    drawn_sigma = rng.random(size=len(chosen_agents))
    chosen_agents = chosen_agents[drawn_sigma <= sigma]
    return intervention_gradient(W, F_rm, R_rm, b, n_roleModels, chosen_agents, rng)

def parse_NK(env):
    n_landscapes = int(str(env.split("_")[1]).split(".")[0])
    N, K = str(env.split("_")[0])[1:].split("K")
    N, K = int(N), int(K)
    return N, K, n_landscapes