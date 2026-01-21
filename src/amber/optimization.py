from typing import Type, Dict, Any, List, Callable, Optional, Union, Tuple
import polars as pl
import numpy as np
import itertools
import random
import time
from .model import Model

# SMAC is an optional dependency - lazy import when needed
HAS_SMAC = False
try:
    import smac
    HAS_SMAC = True
except ImportError:
    pass


def _check_smac():
    """Check if SMAC is available, raise helpful error if not."""
    if not HAS_SMAC:
        raise ImportError(
            "SMAC is required for advanced optimization features. "
            "Install it with: pip install smac"
        )


# Simple ParameterSpace for basic optimization functions
class ParameterSpace:
    """Define the parameter space for optimization."""
    
    def __init__(self, parameters: Dict[str, Any]):
        """Initialize parameter space.
        
        Args:
            parameters: Dictionary mapping parameter names to values or ranges
        """
        self.parameters = parameters
        
    def sample(self) -> Dict[str, Any]:
        """Sample a random parameter combination.
        
        Returns:
            Dictionary with parameter values
        """
        from .experiment import IntRange
        
        result = {}
        for name, value in self.parameters.items():
            if isinstance(value, list):
                result[name] = np.random.choice(value)
            elif isinstance(value, IntRange):  # IntRange objects
                result[name] = np.random.randint(value.start, value.end + 1)
            else:  # Fixed value
                result[name] = value
        return result
        
    def grid_sample(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations in a grid.
        
        Returns:
            List of parameter dictionaries
        """
        from .experiment import IntRange
        
        param_lists = {}
        for name, value in self.parameters.items():
            if isinstance(value, list):
                param_lists[name] = value
            elif isinstance(value, IntRange):  # IntRange objects
                param_lists[name] = list(range(value.start, value.end + 1))
            else:  # Fixed value
                param_lists[name] = [value]
        
        # Generate all combinations
        names = list(param_lists.keys())
        combinations = list(itertools.product(*[param_lists[name] for name in names]))
        
        return [dict(zip(names, combo)) for combo in combinations]


def objective_function(model_class: Type[Model], parameters: Dict[str, Any], 
                      metric: str, iterations: int = 1, minimize: bool = False) -> float:
    """Evaluate objective function for a model with given parameters.
    
    Args:
        model_class: Model class to instantiate
        parameters: Parameters to pass to model
        metric: Name of metric to optimize
        iterations: Number of iterations to average over
        minimize: Whether to minimize (True) or maximize (False)
        
    Returns:
        Objective value
    """
    total = 0.0
    
    for _ in range(iterations):
        # Disable progress reporting for optimization
        model_params = parameters.copy()
        model_params['show_progress'] = False
        model = model_class(model_params)
        results = model.run()
        
        # Get the metric value from model data
        model_data = results['model']
        if metric in model_data.columns:
            # Get the last recorded value of the metric
            values = model_data[metric].to_list()
            if values:
                value = values[-1]
            else:
                value = 0
        else:
            value = 0
            
        total += value
    
    average = total / iterations
    return average


def grid_search(model_class: Type[Model], parameter_space: ParameterSpace, 
                metric: str, iterations: int = 1, minimize: bool = False) -> List[Dict[str, Any]]:
    """Perform grid search optimization.
    
    Args:
        model_class: Model class to optimize
        parameter_space: Parameter space to search
        metric: Metric to optimize
        iterations: Number of iterations per parameter combination
        minimize: Whether to minimize the metric
        
    Returns:
        List of results sorted by objective value (best first)
    """
    results = []
    
    for params in parameter_space.grid_sample():
        obj_value = objective_function(model_class, params, metric, iterations, minimize)
        results.append({
            'parameters': params,
            'objective': obj_value
        })
    
    # Sort by objective value (descending for maximization, ascending for minimization)
    results.sort(key=lambda x: x['objective'], reverse=not minimize)
    
    return results


def random_search(model_class: Type[Model], parameter_space: ParameterSpace, 
                  metric: str, n_samples: int = 10, iterations: int = 1, 
                  minimize: bool = False, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Perform random search optimization.
    
    Args:
        model_class: Model class to optimize
        parameter_space: Parameter space to search
        metric: Metric to optimize
        n_samples: Number of random samples to evaluate
        iterations: Number of iterations per parameter combination
        minimize: Whether to minimize the metric
        seed: Random seed for reproducibility
        
    Returns:
        List of results sorted by objective value (best first)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    results = []
    
    for _ in range(n_samples):
        params = parameter_space.sample()
        obj_value = objective_function(model_class, params, metric, iterations, minimize)
        results.append({
            'parameters': params,
            'objective': obj_value
        })
    
    # Sort by objective value
    results.sort(key=lambda x: x['objective'], reverse=not minimize)
    
    return results


def bayesian_optimization(model_class: Type[Model], parameter_space: ParameterSpace, 
                         metric: str, n_calls: int = 10, iterations: int = 1, 
                         minimize: bool = False, random_state: Optional[int] = None) -> List[Dict[str, Any]]:
    """Perform Bayesian optimization using simple heuristics.
    
    Note: This is a simplified implementation for testing purposes.
    For production use, consider using the SMACOptimizer class.
    
    Args:
        model_class: Model class to optimize
        parameter_space: Parameter space to search
        metric: Metric to optimize
        n_calls: Number of function evaluations
        iterations: Number of iterations per parameter combination
        minimize: Whether to minimize the metric
        random_state: Random state for reproducibility
        
    Returns:
        List of results sorted by objective value (best first)
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    results = []
    
    # Start with random samples (simplified acquisition)
    for _ in range(n_calls):
        params = parameter_space.sample()
        obj_value = objective_function(model_class, params, metric, iterations, minimize)
        results.append({
            'parameters': params,
            'objective': obj_value
        })
    
    # Sort by objective value
    results.sort(key=lambda x: x['objective'], reverse=not minimize)
    
    return results


# Advanced SMAC-based ParameterSpace for complex optimization
class SMACParameterSpace:
    """Define the parameter space for SMAC optimization."""
    
    def __init__(self):
        """Initialize parameter space."""
        self.parameters = {}
        self.fidelity_parameters = {}
        
    def add_parameter(self, name: str, param_type: str, 
                     bounds: Optional[tuple] = None,
                     choices: Optional[List[Any]] = None,
                     default: Any = None,
                     is_fidelity: bool = False):
        """Add a parameter to the space.
        
        Args:
            name: Parameter name
            param_type: Type of parameter ('float', 'int', 'categorical')
            bounds: Tuple of (min, max) for numeric parameters
            choices: List of possible values for categorical parameters
            default: Default value
            is_fidelity: Whether this is a fidelity parameter
        """
        if param_type not in ['float', 'int', 'categorical']:
            raise ValueError("param_type must be 'float', 'int', or 'categorical'")
            
        if param_type in ['float', 'int'] and bounds is None:
            raise ValueError(f"bounds must be provided for {param_type} parameters")
            
        if param_type == 'categorical' and choices is None:
            raise ValueError("choices must be provided for categorical parameters")
            
        param_dict = {
            'type': param_type,
            'bounds': bounds,
            'choices': choices,
            'default': default
        }
        
        if is_fidelity:
            self.fidelity_parameters[name] = param_dict
        else:
            self.parameters[name] = param_dict
            
    def get_configspace(self):
        """Get the SMAC configuration space."""
        from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, \
            UniformIntegerHyperparameter, CategoricalHyperparameter
            
        cs = ConfigurationSpace()
        
        # Add regular parameters
        for name, param in self.parameters.items():
            if param['type'] == 'float':
                hp = UniformFloatHyperparameter(
                    name=name,
                    lower=param['bounds'][0],
                    upper=param['bounds'][1],
                    default_value=param['default']
                )
            elif param['type'] == 'int':
                hp = UniformIntegerHyperparameter(
                    name=name,
                    lower=param['bounds'][0],
                    upper=param['bounds'][1],
                    default_value=param['default']
                )
            else:  # categorical
                hp = CategoricalHyperparameter(
                    name=name,
                    choices=param['choices'],
                    default_value=param['default']
                )
            cs.add_hyperparameter(hp)
            
        # Add fidelity parameters
        for name, param in self.fidelity_parameters.items():
            if param['type'] == 'float':
                hp = UniformFloatHyperparameter(
                    name=name,
                    lower=param['bounds'][0],
                    upper=param['bounds'][1],
                    default_value=param['default']
                )
            elif param['type'] == 'int':
                hp = UniformIntegerHyperparameter(
                    name=name,
                    lower=param['bounds'][0],
                    upper=param['bounds'][1],
                    default_value=param['default']
                )
            else:  # categorical
                hp = CategoricalHyperparameter(
                    name=name,
                    choices=param['choices'],
                    default_value=param['default']
                )
            cs.add_hyperparameter(hp)
            
        return cs

class SMACOptimizer:
    """Optimize model parameters using SMAC with various strategies."""
    
    def __init__(self, model_type: Type[Model], 
                 param_space: SMACParameterSpace,
                 objective: Callable[[Model], float],
                 n_trials: int = 100,
                 n_workers: int = 1,
                 seed: Optional[int] = None,
                 strategy: str = 'bayesian',
                 acquisition_function: str = 'ei',
                 initial_design: str = 'latin_hypercube',
                 surrogate_model: str = 'random_forest',
                 use_multi_fidelity: bool = False,
                 use_random_search: bool = False):
        """Initialize the optimizer.
        
        Args:
            model_type: Class of model to optimize
            param_space: Parameter space definition
            objective: Function that takes a model and returns a score to minimize
            n_trials: Number of optimization trials
            n_workers: Number of parallel workers
            seed: Random seed
            strategy: Optimization strategy ('bayesian', 'random', 'algorithm_configuration')
            acquisition_function: Acquisition function ('ei', 'lcb', 'pi', 'eips', 'log_ei')
            initial_design: Initial design strategy ('latin_hypercube', 'random', 'sobol')
            surrogate_model: Surrogate model type ('random_forest', 'gaussian_process', 'random_forest_with_instances')
            use_multi_fidelity: Whether to use multi-fidelity optimization
            use_random_search: Whether to use random search
        """
        # Check SMAC availability and do lazy imports
        _check_smac()
        from smac import HyperparameterOptimizationFacade, Scenario, MultiFidelityFacade, RandomFacade, AlgorithmConfigurationFacade
        from smac.model.random_forest import RandomForest
        from smac.model.gaussian_process import GaussianProcess
        from smac.acquisition.function import EI, LCB, PI, EIPS, TS
        from smac.acquisition.maximizer import LocalAndSortedRandomSearch
        from smac.initial_design import LatinHypercubeInitialDesign, RandomInitialDesign, SobolInitialDesign
        from smac.intensifier import SuccessiveHalving
        
        self.model_type = model_type
        self.param_space = param_space
        self.objective = objective
        self.n_trials = n_trials
        self.n_workers = n_workers
        self.seed = seed
        
        # Initialize SMAC components
        self.configspace = param_space.get_configspace()
        
        # Select initial design
        if initial_design == 'latin_hypercube':
            initial_design = LatinHypercubeInitialDesign
        elif initial_design == 'random':
            initial_design = RandomInitialDesign
        elif initial_design == 'sobol':
            initial_design = SobolInitialDesign
        else:
            raise ValueError(f"Unknown initial design: {initial_design}")
            
        # Select acquisition function
        if acquisition_function == 'ei':
            acq_func = EI()
        elif acquisition_function == 'lcb':
            acq_func = LCB()
        elif acquisition_function == 'pi':
            acq_func = PI()
        elif acquisition_function == 'eips':
            acq_func = EIPS()
        elif acquisition_function == 'log_ei':
            acq_func = TS()
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")
            
        # Select surrogate model
        if surrogate_model == 'random_forest':
            model = RandomForest()
        elif surrogate_model == 'gaussian_process':
            model = GaussianProcess()
        else:
            raise ValueError(f"Unknown model type: {surrogate_model}")
            
        # Create scenario
        self.scenario = Scenario(
            self.configspace,
            n_trials=n_trials,
            n_workers=n_workers,
            seed=seed
        )
        
        # Initialize appropriate SMAC facade
        if use_multi_fidelity:
            if not param_space.fidelity_parameters:
                raise ValueError("No fidelity parameters defined for multi-fidelity optimization")
            self.smac = MultiFidelityFacade(
                scenario=self.scenario,
                target_function=self._evaluate_config,
                acquisition_function=acq_func,
                model=model,
                initial_design=initial_design(
                    scenario=self.scenario,
                    n_configs=min(10, n_trials)
                ),
                intensifier=SuccessiveHalving(
                    scenario=self.scenario,
                    incumbent_selection="highest_budget",
                    max_incumbents=1
                )
            )
        elif use_random_search:
            self.smac = RandomFacade(
                scenario=self.scenario,
                target_function=self._evaluate_config
            )
        elif strategy == 'algorithm_configuration':
            self.smac = AlgorithmConfigurationFacade(
                scenario=self.scenario,
                target_function=self._evaluate_config,
                acquisition_function=acq_func,
                model=model,
                initial_design=initial_design(
                    scenario=self.scenario,
                    n_configs=min(10, n_trials)
                )
            )
        else:  # bayesian
            self.smac = HyperparameterOptimizationFacade(
                scenario=self.scenario,
                target_function=self._evaluate_config,
                acquisition_function=acq_func,
                model=model,
                initial_design=initial_design(
                    scenario=self.scenario,
                    n_configs=min(10, n_trials)
                ),
                acquisition_maximizer=LocalAndSortedRandomSearch(
                    configspace=self.configspace,
                    acquisition_function=acq_func,
                    challengers=1000,
                    local_search_iterations=10
                )
            )
            
    def _evaluate_config(self, config: Dict[str, Any]) -> float:
        """Evaluate a parameter configuration.
        
        Args:
            config: Parameter configuration
            
        Returns:
            Objective value
        """
        # Create and run model with configuration
        model = self.model_type(config)
        results = model.run()
        
        # Calculate objective value
        return self.objective(model)
        
    def optimize(self) -> Dict[str, Any]:
        """Run the optimization.
        
        Returns:
            Dictionary containing best configuration and results
        """
        # Run optimization
        incumbent = self.smac.optimize()
        
        # Get optimization history
        history = self.smac.runhistory
        
        # Convert history to DataFrame
        data = []
        for config_id, run_value in history.data.items():
            config = history.ids_config[config_id]
            data.append({
                **config.get_dictionary(),
                'cost': run_value.cost,
                'time': run_value.time
            })
            
        history_df = pl.DataFrame(data)
        
        return {
            'best_config': incumbent.get_dictionary(),
            'best_cost': history.get_cost(incumbent),
            'history': history_df
        }

class MultiObjectiveSMAC:
    """Multi-objective parameter optimization using SMAC."""
    
    def __init__(self, model_type: Type[Model],
                 param_space: SMACParameterSpace,
                 objectives: Dict[str, Callable[[Model], float]],
                 n_trials: int = 100,
                 n_workers: int = 1,
                 seed: Optional[int] = None,
                 strategy: str = 'pareto',
                 use_multi_fidelity: bool = False):
        """Initialize the multi-objective optimizer.
        
        Args:
            model_type: Class of model to optimize
            param_space: Parameter space definition
            objectives: Dictionary of objective functions
            n_trials: Number of optimization trials
            n_workers: Number of parallel workers
            seed: Random seed
            strategy: Multi-objective strategy ('pareto', 'aggregation')
            use_multi_fidelity: Whether to use multi-fidelity optimization
        """
        # Check SMAC availability and do lazy imports
        _check_smac()
        from smac import HyperparameterOptimizationFacade, Scenario, MultiFidelityFacade
        from smac.intensifier import SuccessiveHalving
        from smac.multi_objective import AbstractMultiObjectiveAlgorithm
        from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
        
        self.model_type = model_type
        self.param_space = param_space
        self.objectives = objectives
        self.n_trials = n_trials
        self.n_workers = n_workers
        self.seed = seed
        
        # Initialize SMAC components
        self.configspace = param_space.get_configspace()
        self.scenario = Scenario(
            self.configspace,
            n_trials=n_trials,
            n_workers=n_workers,
            seed=seed
        )
        
        # Initialize multi-objective algorithm
        if strategy == 'aggregation':
            mo_algorithm = MeanAggregationStrategy(
                scenario=self.scenario,
                objectives=list(objectives.keys())
            )
        else:  # pareto
            mo_algorithm = AbstractMultiObjectiveAlgorithm(
                scenario=self.scenario,
                objectives=list(objectives.keys())
            )
            
        # Initialize SMAC facade for each objective
        self.smacs = {}
        for name, objective in objectives.items():
            if use_multi_fidelity:
                if not param_space.fidelity_parameters:
                    raise ValueError("No fidelity parameters defined for multi-fidelity optimization")
                self.smacs[name] = MultiFidelityFacade(
                    scenario=self.scenario,
                    target_function=lambda config: self._evaluate_objective(config, objective),
                    multi_objective_algorithm=mo_algorithm,
                    intensifier=SuccessiveHalving(
                        scenario=self.scenario,
                        incumbent_selection="highest_budget",
                        max_incumbents=1
                    )
                )
            else:
                self.smacs[name] = HyperparameterOptimizationFacade(
                    scenario=self.scenario,
                    target_function=lambda config: self._evaluate_objective(config, objective),
                    multi_objective_algorithm=mo_algorithm
                )
                
    def _evaluate_objective(self, config: Dict[str, Any], 
                          objective: Callable[[Model], float]) -> float:
        """Evaluate a parameter configuration for a specific objective.
        
        Args:
            config: Parameter configuration
            objective: Objective function
            
        Returns:
            Objective value
        """
        model = self.model_type(config)
        results = model.run()
        return objective(model)
        
    def optimize(self) -> Dict[str, Any]:
        """Run the multi-objective optimization.
        
        Returns:
            Dictionary containing Pareto-optimal configurations and results
        """
        # Run optimization for each objective
        results = {}
        for name, smac in self.smacs.items():
            incumbent = smac.optimize()
            results[name] = {
                'best_config': incumbent.get_dictionary(),
                'best_cost': smac.runhistory.get_cost(incumbent)
            }
            
        # Get full optimization history
        history = {}
        for name, smac in self.smacs.items():
            data = []
            for config_id, run_value in smac.runhistory.data.items():
                config = smac.runhistory.ids_config[config_id]
                data.append({
                    **config.get_dictionary(),
                    f'{name}_cost': run_value.cost,
                    f'{name}_time': run_value.time
                })
            history[name] = pl.DataFrame(data)
            
        # Find Pareto-optimal configurations
        pareto_front = self._find_pareto_front(history)
        
        return {
            'single_objective_results': results,
            'pareto_front': pareto_front,
            'history': history
        }
        
    def _find_pareto_front(self, history: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Find Pareto-optimal configurations.
        
        Args:
            history: Dictionary of optimization histories
            
        Returns:
            DataFrame containing Pareto-optimal configurations
        """
        from smac.utils.pareto_front import calculate_pareto_front
        from smac.utils.multi_objective import normalize_costs
        
        # Combine all histories
        combined = history[list(history.keys())[0]]
        for name, df in list(history.items())[1:]:
            combined = combined.join(
                df.select(['id', f'{name}_cost']),
                on='id'
            )
            
        # Normalize costs
        costs = np.array([[row[f'{name}_cost'] for name in self.objectives.keys()]
                         for row in combined.iter_rows(named=True)])
        normalized_costs = normalize_costs(costs)
        
        # Find Pareto-optimal points
        pareto_mask = calculate_pareto_front(normalized_costs)
        
        return combined.filter(pl.Series(pareto_mask)) 