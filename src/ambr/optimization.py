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
        
    def sample(self, rng=None) -> Dict[str, Any]:
        """Sample a random parameter combination.

        Args:
            rng: optional ``numpy.random.Generator`` for reproducibility. When
                omitted, a fresh local generator is used (never the global
                ``np.random`` state).

        Returns:
            Dictionary with parameter values
        """
        from .experiment import IntRange

        if rng is None:
            rng = np.random.default_rng()
        result = {}
        for name, value in self.parameters.items():
            if isinstance(value, list):
                result[name] = value[int(rng.integers(0, len(value)))]
            elif isinstance(value, IntRange):  # IntRange objects (exclusive end)
                result[name] = int(rng.integers(value.start, value.end))
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
            elif isinstance(value, IntRange):  # IntRange objects (exclusive end)
                param_lists[name] = list(range(value.start, value.end))
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
    rng = np.random.default_rng(seed)  # local generator; never touches global np.random

    results = []

    for _ in range(n_samples):
        params = parameter_space.sample(rng)
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
                         minimize: bool = False, random_state: Optional[int] = None,
                         n_initial_design: int = 5) -> List[Dict[str, Any]]:
    """Perform Bayesian optimisation using SMAC3's Gaussian Process facade.

    Converts the simple ``ParameterSpace`` to a SMAC3 ``ConfigurationSpace``
    internally and runs true Bayesian optimisation with Expected Improvement
    acquisition. Requires SMAC3 to be installed (``pip install smac``).

    Args:
        model_class: Model class to optimize.
        parameter_space: Parameter space to search.
        metric: Metric to optimize.
        n_calls: Total number of function evaluations.
        iterations: Number of iterations per parameter combination.
        minimize: Whether to minimize (True) or maximize (False).
        random_state: Random state for reproducibility.
        n_initial_design: Number of initial random designs before
            Bayesian search begins.

    Returns:
        List of results sorted by objective value (best first).
    """
    _check_smac()

    from ConfigSpace import (
        ConfigurationSpace,
        UniformIntegerHyperparameter,
        UniformFloatHyperparameter,
        CategoricalHyperparameter,
    )
    import tempfile
    from smac import HyperparameterOptimizationFacade, Scenario
    from smac.acquisition.function import EI
    from smac.model.random_forest import RandomForest
    from smac.initial_design import LatinHypercubeInitialDesign

    from .experiment import IntRange

    # --- build ConfigurationSpace from ParameterSpace -----------------------
    cs = ConfigurationSpace(seed=random_state or 0)
    cat_params: Dict[str, List[Any]] = {}
    fixed_params: Dict[str, Any] = {}

    for name, value in parameter_space.parameters.items():
        if isinstance(value, list):
            # Categorical — SMAC requires strings
            str_choices = [str(v) for v in value]
            cat_params[name] = value  # keep original mapping
            hp = CategoricalHyperparameter(
                name=name,
                choices=str_choices,
                default_value=str_choices[0],
            )
            cs.add(hp)
        elif isinstance(value, IntRange):
            hp = UniformIntegerHyperparameter(
                name=name,
                lower=value.start,
                upper=value.end - 1,  # IntRange.end is exclusive
                default_value=value.start,
            )
            cs.add(hp)
        elif isinstance(value, float):
            hp = UniformFloatHyperparameter(
                name=name,
                lower=value,
                upper=value,
                default_value=value,
            )
            cs.add(hp)
        elif isinstance(value, (int, str, bool)):
            # Fixed scalar — not optimised; stored separately.
            fixed_params[name] = value
        else:
            raise TypeError(
                f"Unsupported parameter type for {name!r}: {type(value)}"
            )

    # --- scenario & SMAC facade --------------------------------------------
    # Use a temporary output directory so each call starts fresh.
    tmp_dir = tempfile.mkdtemp(prefix='amber_bayes_')
    scenario = Scenario(
        cs,
        n_trials=n_calls,
        seed=random_state,
        deterministic=True,
        output_directory=tmp_dir,
    )

    # Target function for SMAC (always minimises)
    def _target(config: dict, seed: int = 0) -> float:
        # Merge SMAC config + fixed params + restore categorical types
        params = dict(fixed_params)
        for k, v in config.items():
            if k in cat_params:
                str_choices = [str(cv) for cv in cat_params[k]]
                try:
                    idx = str_choices.index(str(v))
                    params[k] = cat_params[k][idx]
                except ValueError:
                    params[k] = v
            else:
                params[k] = v
        obj = objective_function(model_class, params, metric, iterations, minimize)
        # SMAC always minimises; if the user wants to maximise, negate
        return -obj if not minimize else obj

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=_target,
        model=RandomForest(configspace=cs),
        acquisition_function=EI(),
        initial_design=LatinHypercubeInitialDesign(
            scenario=scenario,
            n_configs=min(n_initial_design, n_calls),
        ),
    )

    try:
        incumbent = smac.optimize()
    except Exception:
        # Configuration space exhausted — proceed with whatever SMAC3
        # evaluated so far (runhistory still has the partial results).
        pass

    # --- collect history ---------------------------------------------------
    results: List[Dict[str, Any]] = []
    for config in smac.runhistory.get_configs():
        try:
            cost = smac.runhistory.get_cost(config)
        except Exception:
            cost = float('inf')
        params = dict(fixed_params)
        for k, v in dict(config).items():
            if k in cat_params:
                str_choices = [str(cv) for cv in cat_params[k]]
                try:
                    idx = str_choices.index(str(v))
                    params[k] = cat_params[k][idx]
                except ValueError:
                    params[k] = v
            else:
                params[k] = v
        results.append({
            'parameters': params,
            'objective': -cost if not minimize else cost,
        })

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
            
        # Select surrogate model. SMAC 2.x surrogate models require the
        # configuration space as their first argument.
        if surrogate_model == 'random_forest':
            model = RandomForest(self.configspace)
        elif surrogate_model == 'gaussian_process':
            model = GaussianProcess(self.configspace)
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
            
    def _evaluate_config(self, config, seed: int = 0, budget=None) -> float:
        """Evaluate a parameter configuration (SMAC 2.x target function).

        SMAC passes a ``Configuration`` plus a per-trial ``seed`` (and an
        optional ``budget`` for multi-fidelity); both are accepted so SMAC's
        ``TargetFunctionRunner`` can bind them. The seed is injected into the
        model parameters so each trial is reproducible.

        Args:
            config: SMAC ``Configuration`` (dict-like) of parameter values
            seed: per-trial seed supplied by SMAC
            budget: optional fidelity budget (unused in the single-fidelity path)

        Returns:
            Objective value
        """
        params = dict(config)
        params.setdefault('seed', seed)
        params.setdefault('show_progress', False)
        model = self.model_type(params)
        # Store results on the model so a Callable[[Model], float] objective can
        # read ``model.results`` (Model.run returns the dict but does not assign
        # it). SMAC's surrogate cannot be fit on non-finite targets, so map a
        # failed / degenerate evaluation to a large finite penalty.
        try:
            model.results = model.run()
            value = self.objective(model)
        except Exception:
            return 1e10
        if value is None or not np.isfinite(value):
            return 1e10
        return float(value)
        
    def optimize(self) -> Dict[str, Any]:
        """Run the optimization.
        
        Returns:
            Dictionary containing best configuration and results
        """
        # Run optimization
        incumbent = self.smac.optimize()

        # Convert the run history to a DataFrame (SMAC 2.x RunHistory is a
        # Mapping of TrialKey -> TrialValue; access configs via get_config).
        history = self.smac.runhistory
        data = []
        try:
            for trial_key, trial_value in history.items():
                config = history.get_config(trial_key.config_id)
                data.append({
                    **dict(config),
                    'cost': trial_value.cost,
                    'time': getattr(trial_value, 'time', 0.0),
                })
        except Exception:
            data = []
        history_df = pl.DataFrame(data) if data else pl.DataFrame()

        best_cost = history.get_cost(incumbent) if incumbent is not None else None
        return {
            'best_config': dict(incumbent) if incumbent is not None else {},
            'best_cost': best_cost,
            'best_objective': best_cost,  # alias for callers expecting this key
            'history': history_df,
        }

class MultiObjectiveSMAC:
    """Multi-objective optimization by running one SMAC search per objective.

    Each named objective is optimized independently with a scalar SMAC facade
    (same evaluate path as :class:`SMACOptimizer`). Histories are merged and a
    simple non-dominated set is returned as ``pareto_front``. Suitable for
    small multi-objective calibration examples; not a full ParEGO/EHVI MOBO.
    """

    def __init__(
        self,
        model_type: Type[Model],
        param_space: SMACParameterSpace,
        objectives: Dict[str, Callable[[Model], float]],
        n_trials: int = 100,
        n_workers: int = 1,
        seed: Optional[int] = None,
        strategy: str = "pareto",
        use_multi_fidelity: bool = False,
    ):
        _check_smac()
        if use_multi_fidelity:
            raise NotImplementedError(
                "MultiObjectiveSMAC multi-fidelity is not supported; "
                "use SMACOptimizer(use_multi_fidelity=True) per objective."
            )
        if not objectives:
            raise ValueError("objectives must be a non-empty dict")

        self.model_type = model_type
        self.param_space = param_space
        self.objectives = dict(objectives)
        self.n_trials = int(n_trials)
        self.n_workers = n_workers
        self.seed = seed
        self.strategy = strategy
        # Built on first optimize() so construction stays cheap for smoke tests.
        self._optimizers: Optional[Dict[str, SMACOptimizer]] = None

    def _ensure_optimizers(self) -> Dict[str, "SMACOptimizer"]:
        if self._optimizers is not None:
            return self._optimizers
        opts: Dict[str, SMACOptimizer] = {}
        for i, (name, objective) in enumerate(self.objectives.items()):
            # Distinct seeds so independent searches explore differently.
            seed = None if self.seed is None else int(self.seed) + i * 17
            opts[name] = SMACOptimizer(
                model_type=self.model_type,
                param_space=self.param_space,
                objective=objective,
                n_trials=self.n_trials,
                n_workers=self.n_workers,
                seed=seed,
                strategy="bayesian",
            )
        self._optimizers = opts
        return opts

    def optimize(self) -> Dict[str, Any]:
        """Run per-objective SMAC and assemble a Pareto set.

        Returns:
            ``n_evaluations``, ``pareto_front`` (configs + objective costs),
            ``history`` (long-format rows), ``single_objective_results``.
        """
        optimizers = self._ensure_optimizers()
        single: Dict[str, Any] = {}
        history_frames: Dict[str, pl.DataFrame] = {}
        n_evals = 0

        for name, opt in optimizers.items():
            result = opt.optimize()
            single[name] = {
                "best_config": result.get("best_config", {}),
                "best_cost": result.get("best_cost"),
            }
            hist = result.get("history")
            if hist is not None and not hist.is_empty() and "cost" in hist.columns:
                history_frames[name] = hist.rename({"cost": name})
                n_evals += hist.height
            elif result.get("best_config"):
                # Degenerate history: still record the incumbent.
                history_frames[name] = pl.DataFrame(
                    [{**result["best_config"], name: result.get("best_cost")}]
                )
                n_evals += 1

        history_long = (
            pl.concat(list(history_frames.values()), how="diagonal_relaxed")
            if history_frames
            else pl.DataFrame()
        )
        pareto = self._pareto_from_incumbents(single)

        return {
            "n_evaluations": n_evals,
            "pareto_front": pareto,
            "history": history_long if not history_long.is_empty() else history_frames,
            "single_objective_results": single,
        }

    def _pareto_from_incumbents(self, single: Dict[str, Any]) -> pl.DataFrame:
        """Build a small Pareto table from per-objective incumbents.

        Each incumbent is re-scored on *all* objectives so the front carries
        comparable columns for plotting / example code.
        """
        rows = []
        seen = set()
        for _name, res in single.items():
            cfg = dict(res.get("best_config") or {})
            key = tuple(sorted(cfg.items()))
            if key in seen:
                continue
            seen.add(key)
            params = {**cfg, "show_progress": False}
            try:
                model = self.model_type(params)
                model.results = model.run()
                row = {**cfg}
                for obj_name, obj_fn in self.objectives.items():
                    try:
                        row[obj_name] = float(obj_fn(model))
                    except Exception:
                        row[obj_name] = float("inf")
                rows.append(row)
            except Exception:
                continue

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)
        obj_names = list(self.objectives.keys())
        costs = df.select(obj_names).to_numpy()
        # Non-dominated: minimize all objectives.
        n = costs.shape[0]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                    keep[i] = False
                    break
        return df.filter(pl.Series(keep.tolist()))
