from typing import Type, Dict, Any, List, Optional
import polars as pl
from .model import Model
from ._deprecation import warn_deprecated

class IntRange:
    """Range of integer values for parameter sampling.

    Semantics follow Python's ``range()``: ``start`` is inclusive,
    ``end`` is exclusive (i.e. ``IntRange(1, 10)`` yields values
    ``1..9``).
    """

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def __contains__(self, value: int) -> bool:
        return self.start <= value < self.end

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"IntRange({self.start}, {self.end})"

class Sample:
    """Container for parameter combinations.

    **Not a Cartesian product and not independent random sampling.**
    For each index ``i`` in ``0 .. n-1``:

    * scalar values are copied into every combination;
    * list values are taken as ``list[i % len(list)]`` (lists stay
      **index-aligned / zipped**, not crossed);
    * :class:`IntRange` values are spread deterministically across
      ``[start, end)`` (midpoint when ``n == 1``).

    For a full grid use :func:`~ambr.grid_search`; for independent random
    draws use :func:`~ambr.random_search`.
    """

    def __init__(self, parameters: Dict[str, Any], n: int):
        """Initialize a new parameter sample.

        Args:
            parameters: Dictionary of parameters and their ranges
            n: Number of samples to generate
        """
        self.parameters = parameters
        self.n = n
        self.combinations = self._generate_combinations()

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations (zip/cycle semantics; see class doc)."""
        if self.n == 0:
            return []

        combinations = []
        ranges = []
        lists = []
        fixed = {}

        # Separate different parameter types
        for key, value in self.parameters.items():
            if isinstance(value, IntRange):
                ranges.append((key, value))
            elif isinstance(value, list):
                lists.append((key, value))
            else:
                fixed[key] = value

        # Generate n combinations
        for i in range(self.n):
            combo = fixed.copy()

            # Handle IntRange parameters (end is exclusive)
            for key, range_obj in ranges:
                n_values = range_obj.end - range_obj.start  # e.g. 10..20 → 10 values
                if self.n == 1:
                    # If only one sample, use middle value
                    value = (range_obj.start + range_obj.end - 1) // 2
                else:
                    # Distribute evenly across [start, end-1]
                    step = (n_values - 1) / (self.n - 1)
                    value = int(range_obj.start + round(i * step))
                combo[key] = value

            # Handle list parameters (cycle through values)
            for key, value_list in lists:
                combo[key] = value_list[i % len(value_list)]

            combinations.append(combo)

        return combinations

class Experiment:
    """Run many parameter combinations of one model class.

    Canonical constructor::

        Experiment(model_type=MyModel, sample=Sample({...}, n=10), iterations=1)

    Legacy aliases (deprecated → 1.0)::

        Experiment(model_class=MyModel, parameters=sample, iterations=1)

    Returns a dict with ``parameters`` / ``agents`` / ``model`` as Polars
    frames and ``info`` as a **Python dict** (not a frame) — **not** a
    pandas object and **not** automatic multi-process parallelism. For CPU
    process pools use :class:`~ambr.performance.ParallelRunner`; for many
    short GPU replicates use :class:`~ambr.gpu_ensemble.GPUEnsembleRunner`.
    """

    def __init__(
        self,
        model_type: Optional[Type[Model]] = None,
        sample: Optional[Sample] = None,
        iterations: int = 1,
        record: bool = True,
        *,
        model_class: Optional[Type[Model]] = None,
        parameters: Optional[Sample] = None,
    ):
        """Initialize a new experiment.

        Args:
            model_type: Class of model to run (canonical).
            sample: :class:`Sample` of parameter combinations (canonical).
            iterations: Number of iterations per parameter combination.
            record: Reserved for future use (results always include frames).
            model_class: Deprecated alias for ``model_type``.
            parameters: Deprecated alias for ``sample``.
        """
        if model_class is not None:
            warn_deprecated(
                "Experiment(model_class=...)",
                "Experiment(model_type=...)",
            )
            if model_type is None:
                model_type = model_class
        if parameters is not None:
            warn_deprecated(
                "Experiment(parameters=...)",
                "Experiment(sample=...)",
            )
            if sample is None:
                sample = parameters
        if model_type is None:
            raise TypeError("Experiment requires model_type= (model class)")
        if sample is None:
            raise TypeError("Experiment requires sample= (a Sample instance)")
        if not isinstance(sample, Sample):
            raise TypeError(
                f"sample must be a Sample instance, got {type(sample).__name__}"
            )
        self.model_type = model_type
        self.sample = sample
        self.iterations = iterations
        self.record = record

    def run(self) -> Dict[str, Any]:
        """Run the experiment.

        Returns:
            Dictionary with ``info``, ``parameters``, ``agents``, ``model``
            (Polars DataFrames for the table keys).
        """
        all_results = []
        all_agents_data = []
        all_model_data = []

        # Run simulations for each parameter combination
        for params in self.sample.combinations:
            for i in range(self.iterations):
                # Add iteration number to parameters
                run_params = params.copy()
                if self.iterations > 1:
                    run_params['iteration'] = i

                # Disable progress reporting for experiments
                run_params['show_progress'] = False

                # Run simulation
                model = self.model_type(run_params)
                results = model.run()
                all_results.append(results)

                # Add parameter information to agent data
                if len(results['agents']) > 0:
                    agents_with_params = results['agents'].with_columns([
                        pl.lit(params[k]).alias(k) for k in params.keys()
                    ])
                    if self.iterations > 1:
                        agents_with_params = agents_with_params.with_columns([
                            pl.lit(i).alias('iteration')
                        ])
                    all_agents_data.append(agents_with_params)

                # Add parameter information to model data
                model_with_params = results['model'].with_columns([
                    pl.lit(params[k]).alias(k) for k in params.keys()
                ])
                if self.iterations > 1:
                    model_with_params = model_with_params.with_columns([
                        pl.lit(i).alias('iteration')
                    ])
                all_model_data.append(model_with_params)

        # Combine results
        combined = {
            'info': {
                'model_type': self.model_type.__name__,
                'sample_size': len(self.sample.combinations),
                'iterations': self.iterations
            },
            'parameters': pl.DataFrame(self.sample.combinations),
            'agents': pl.concat(all_agents_data) if all_agents_data else pl.DataFrame(),
            'model': pl.concat(all_model_data) if all_model_data else pl.DataFrame()
        }

        return combined
