import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy import interpolate
from itertools import product

"""
**Environments** 
(Avg Payoff (PO) Hybrid Agents 2016 Paper):

- MasonWatts 
    - [R code](https://github.com/dnlbrkc/collective_search/blob/master/environments/functions.R)

"""

class Environment():

    def get_fitness(solution):
        raise NotImplementedError
    
    def get_all_values():
        raise NotImplementedError

    def get_title():
        raise NotImplementedError
    
class Environment2D(Environment):
    
    def __init__(self, env_type, grid_size=1000):
         # Fitness landscapes
        if env_type.lower() == "ackley":
            fitness_m = self.ackley(grid_size); self.env_title = "Ackley"
        elif env_type.lower() == "masonwatts":
            fitness_m = self.mason_watts(grid_size); self.env_title = "MasonWatts"
        elif env_type.lower() == "crossintray":
            fitness_m = self.cross_in_tray(grid_size); self.env_title = "CrossInTray"
        elif env_type.lower() == "shubert":
            fitness_m = self.shubert(grid_size); self.env_title = "Shubert"
        elif env_type.lower() == "dropwave":
            fitness_m = self.drop_wave(grid_size); self.env_title = "DropWave"
        elif env_type.lower() == "schaffer":
            fitness_m = self.schaffer(grid_size); self.env_title = "SchafferN4"
        elif env_type.lower() == "table":
            fitness_m = self.holder_table(grid_size); self.env_title = "HolderTable"
        else:
            raise NotImplementedError("Fitness Env not Implemented")
    
        fitness_m -= fitness_m.min()
        self.fitness_m = fitness_m / fitness_m.max()
        self.grid_size = grid_size
    
    def get_fitness(self, solution):
        return self.fitness_m[solution[0], solution[1]]

    def get_title(self):
        return self.env_title
    
    def get_all_values(self):
        return self.fitness_m


    def ackley(self, grid_size=1000, a=20, b=0.2, c=2 * np.pi):
        """
        Create matrix with ackley function between -32.768, 32.768
        @args: Scales of oscillation/ steepness; default values are the most common used
        """
        x = y = np.linspace(-32.768, 32.768, grid_size)
        xx, yy = np.meshgrid(x, y)

        # Wikipedia formula
        sum_1 = 0.5 * (xx ** 2 + yy ** 2)
        sum_2 = np.cos(c * xx) + np.cos(c * yy)

        term1 = -a * np.exp(-b * np.sqrt(sum_1))
        term2 = -np.exp(0.5 * sum_2)

        fitness_m = term1 + term2 + a + np.e

        # Flip and normalize for ascend
        fitness_m = -fitness_m
        fitness_m = fitness_m + (-fitness_m.min())
        fitness_m = fitness_m * (1 / fitness_m.max())
        return fitness_m


    def mason_watts(self, grid_size=1000):
        """One clear max based on gauß + random noise"""
        var = 3 * (grid_size / 100)
        rho = 0.7

        # 1. Generate Random gaussian grid
        x = y = grid_axes = np.linspace(0, grid_size, grid_size)
        mu = [np.random.uniform(1, grid_size), np.random.uniform(1, grid_size)]
        x = norm.pdf(x, loc=mu[0], scale=np.sqrt(var))
        y = norm.pdf(y, loc=mu[1], scale=np.sqrt(var))
        fitness_m = np.outer(x, y)
        fitness_m = fitness_m * (1 / fitness_m.max())  # Note: Not sure if they normalize (taken from 2016 Code)

        # Alternative: Replace by multimodal gaußian with two random means
        # xx, yy = np.meshgrid(x,y)
        # xy = np.column_stack([xx.flat, yy.flat])
        # fitness_m = multivariate_normal.pdf(xy, mean=mu, cov=np.diag([np.sqrt(var), np.sqrt(var)])).reshape(grid_size, grid_size)
        # fitness_m = fitness_m * (1/ fitness_m.max())

        # 2. Random Perlin noise
        for omega in range(3, 8):
            octave = 2 ** omega

            # i) Create smaller grid with uniform values
            octave_m = np.random.uniform(size=octave ** 2).reshape(octave, octave)

            # ii) Use interpolation to smooth the values of the smaller grid across the neighbours
            octave_seq = np.linspace(1, grid_size, octave)

            # bicubic interpolation = interpolation of bivariate function based on third degree polynomial
            # Should be faster, but interpolation does not work smooth: interpBi_m = interpolate.RectBivariateSpline(octave_seq, octave_seq, octave_m, bbox=[min(octave_seq), max(octave_seq), min(octave_seq), max(octave_seq)])
            interp2d_m = interpolate.interp2d(octave_seq, octave_seq, octave_m, kind="cubic")

            # iii) Scale with persistence parameter
            octave_m = interp2d_m(grid_axes, grid_axes) * rho ** omega

            fitness_m = fitness_m + octave_m

        fitness_m = fitness_m * (1 / fitness_m.max())

        return fitness_m

    def cross_in_tray(self, grid_size=1000):
        """Formula from: https://www.sfu.ca/~ssurjano/crossit.html
        and https://www.indusmic.com/post/python-implementation-of-cross-in-tray-function-1"""
        x = y = np.linspace(-10, 10, grid_size)
        xx, yy = np.meshgrid(x, y)

        exp_term = np.abs(100 - np.sqrt(xx ** 2 + yy ** 2) / np.pi)
        main_part = np.abs(np.sin(xx) * np.sin(yy) * np.exp(exp_term)) + 1
        fitness_m = -0.0001 * main_part ** 0.1
        fitness_m = -fitness_m + fitness_m.max()
        fitness_m = fitness_m * (1 / fitness_m.max())
        return fitness_m

    def shubert(self, grid_size=1000):
        """Formula: https://www.sfu.ca/~ssurjano/shubert.html"""
        x = y = np.linspace(-5.12, 5.12, grid_size)
        xx, yy = np.meshgrid(x, y)
        ii = np.arange(1, 6)
        first_term = np.sum([i * np.cos((i+1) * xx + i) for i in ii], axis=0)
        second_term = np.sum([i * np.cos((i+1) * yy + i) for i in ii], axis=0)
        fitness_m = -(first_term * second_term)
        fitness_m -= fitness_m.min() # + (fitness_m.min())
        fitness_m = fitness_m * (1/fitness_m.max())
        return fitness_m

    def drop_wave(self, grid_size=1000):
        x = y = np.linspace(-5.12, 5.12, grid_size)
        xx, yy = np.meshgrid(x, y)
        nominator = 1 + np.cos(12 * np.sqrt(xx ** 2 + yy ** 2))
        divisor = 0.5 * (xx ** 2 + yy ** 2) + 2

        fitness_m = -(nominator / divisor)
        # fitness_m -= fitness_m.min() # + (fitness_m.min())
        fitness_m = fitness_m * (1 / fitness_m.max())
        return fitness_m

    def schaffer(self, grid_size=1000):
        x = y = np.linspace(-50, 50, grid_size)
        xx, yy = np.meshgrid(x, y)
        nominator = np.cos(np.sin(np.abs(xx**2 - yy**2)))**2 - 0.5
        divisor = (1 + 0.001*(xx**2 + yy**2))**2

        fitness_m = (nominator / divisor) # +0.5
        fitness_m = fitness_m * (1 / fitness_m.max())
        return fitness_m

    def holder_table(self, grid_size=1000):
        x = y = np.linspace(-10, 10, grid_size)
        xx, yy = np.meshgrid(x, y)
        fitness_m = np.abs(np.sin(xx)*np.cos(yy)*np.exp(np.abs(1-(np.sqrt(xx**2 + yy**2)/ np.pi))))
        fitness_m = fitness_m * (1 / fitness_m.max())
        return fitness_m


class NK_landscape(Environment):

    def __init__(self, N=10, K=5, seed=None, C=None, dependencies=None):
        """
        Calculates a dictionary with values for every possible state.
        Payoff is scaled form 0-1
        C: Table with values for every combination of connected components
        fitness_dict: keys are the base10 converted states, values the fitness payoff
        """
        assert N >= K + 1

        self.N = N
        self.K = K

        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng(np.random.randint(0, 100))
        if C is None or dependencies is None:
            self.initialize_C()
            self.initialize_dependencies()
        else:
            self.dependencies = dependencies
            self.C = C

        self.combinations = np.array(list(product([0, 1], repeat=K + 1)))
        self.max_value = None
        self.min_value = None

        self.fitness_dict = {}
        self.init_values()

    def initialize_dependencies(self):
        """[:,0] is connected with [:,1] --> [0,1] would mean node 0 has fluence on node 1"""
        depend = np.zeros((self.N, self.K + 1))
        x = np.arange(self.N)
        depend[:, 0] = x
        index = np.arange(self.K) + 1
        self.rng.shuffle(index)
        for _ in range(self.K):
            depend[:, _ + 1] = np.roll(x, index[_])
        self.dependencies = depend.astype(int)

    def initialize_C(self):
        self.C = self.rng.random((self.N, 2 ** (self.K + 1))).round(1)

    def f(self, state):
        """Checks all connected dimensions for the combination they are currently in and adds the value"""
        assert len(state) == self.N
        solution = np.empty(len(state))
        for i in range(len(self.dependencies)):
            solution[i] = self.C[i, int("".join(map(str, state[self.dependencies[i]])), 2)]
        return round(solution.mean(), 2)

    def init_values(self):
        """Creates a dictionary mapping all states to their payoff"""
        max_val = 0
        min_val = 2
        for sol in np.array(list(product([0, 1], repeat=self.N))):
            key = int("".join(map(str, sol)), 2)
            value = self.f(sol)
            self.fitness_dict[key] = value

            if max_val < value:
                max_val = value
            if min_val > value:
                min_val = value
        self.max_value = max_val
        self.min_value = min_val

    def get_fitness(self, state):
        assert len(state) == self.N
        key = int("".join(map(str, state)), 2)
        value = self.fitness_dict[key]
        return np.round((value - self.min_value) / (self.max_value - self.min_value), 2)
    
    def get_title(self):
        return f"N{self.N}K{self.K}"
    
    
class NK_landscape_loaded(NK_landscape):

    def __init__(self, N, K, fitness_dict):
        assert N >= K + 1

        self.N = N
        self.K = K

        self.max_value = None
        self.min_value = None

        self.fitness_dict = fitness_dict
        self.init_min_max()

    def f(self, state):
        raise NotImplementedError("Only dictionary in loaded nk landscape")

    def init_min_max(self):
        self.max_value = max(self.fitness_dict.values())
        self.min_value = min(self.fitness_dict.values())
    
    def get_all_values(self):
        return (self.fitness_dict.values() - self.min_value) / (self.max_value - self.min_value)

