import os
import shutil
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

class SaveModelStrategy(FedAvg):
    def __init__(self,  *args, **kwargs):
        self.score_key = kwargs.pop("score_key", "rouge1")
        self.cfg = kwargs.pop("cfg", {})
        
        super().__init__(*args, **kwargs)
        self.previous_best_score = 0
        self.best_parameters = None
        self.best_metrics = None
        
    def evaluate(self, server_round, parameters):
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        curr_score = metrics[self.score_key]
        checkpoint_dir = os.path.join(self.cfg.output_dir, f"checkpoint-{server_round}")
        if curr_score > self.previous_best_score:
            self.previous_best_score = curr_score
            self.best_metrics = metrics
        else:
            shutil.rmtree(checkpoint_dir)   
        return loss, metrics
