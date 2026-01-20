from typing import Type, Dict, Any, List
import polars as pl
from .model import Model

class IntRange:
    """Range of integer values for parameter sampling."""
    
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        
    def __repr__(self) -> str:
        return f"IntRange({self.start}, {self.end})"
        
class Sample:
    """Container for parameter combinations."""
    
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
        """Generate parameter combinations."""
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
            
            # Handle IntRange parameters
            for key, range_obj in ranges:
                if self.n == 1:
                    # If only one sample, use middle value
                    value = (range_obj.start + range_obj.end) // 2
                else:
                    # Distribute evenly across range
                    step = (range_obj.end - range_obj.start) / (self.n - 1)
                    value = int(range_obj.start + i * step)
                combo[key] = value
            
            # Handle list parameters (cycle through values)
            for key, value_list in lists:
                combo[key] = value_list[i % len(value_list)]
                
            combinations.append(combo)
            
        return combinations

class Experiment:
    """Container for running multiple model simulations."""
    
    def __init__(self, model_type: Type[Model], sample: Sample, 
                 iterations: int = 1, record: bool = True):
        """Initialize a new experiment.
        
        Args:
            model_type: Class of model to run
            sample: Parameter sample to use
            iterations: Number of iterations per parameter combination
            record: Whether to record variables during simulation
        """
        self.model_type = model_type
        self.sample = sample
        self.iterations = iterations
        self.record = record
        
    def run(self) -> Dict[str, Any]:
        """Run the experiment.
        
        Returns:
            Dictionary containing results from all runs
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