"""FedAsync (Federated Asynchronous) federated learning strategy.

Paper: Xie et al, Federated Asynchronous Optimization
https://arxiv.org/pdf/1903.03934.pdf
"""
from .fedavg import FedAvg


class FedAsync(FedAvg):
    """An asynchronous version of Federated Averaging that updates the global
    model upon each client update."""

    def aggregate_fit(
        self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client parameter updates one at a time, asynchronously."""
        #  Input: local_model_before_training, final_local_model, global_model, wt
        # Output: global_model = global_model - lr*delta, where
        #    delta = (local_model_before_training - final_local_model)
        #    lr: original_lr*wt (higher wt = higher learning rate)
        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate client evaluation results one at a time, asynchronously."""
        return super().aggregate_evaluate(rnd, results, failures)()
