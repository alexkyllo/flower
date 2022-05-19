"""FedAsync (Federated Asynchronous) federated learning strategy.

Paper: Xie et al, Federated Asynchronous Optimization
https://arxiv.org/pdf/1903.03934.pdf
"""
from typing import Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateRes,
    Parameters,
    Scalar,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.typing import FitRes, Weights
from flwr.server.client_proxy import ClientProxy

from .fedavg import FedAvg


def update_global_from_local(
    global_model: Weights, local_model: Weights, learning_rate: float
) -> Weights:
    """Update the"""
    # Assume we're trying to aggregate updates from userA
    # local_model = model after local training by userA
    # global_model = model on server when userA finishes training
    # delta = global_model - final_local_model
    # new_global_model = global_model - delta*lr
    delta = global_model - local_model
    result = global_model - learning_rate * delta
    return result


class FedAsync(FedAvg):
    """An asynchronous version of Federated Averaging that updates the global
    model upon each client update."""

    def aggregate_fit(
        self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client parameter updates one at a time, asynchronously."""

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results sent from client to weights
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = weights_to_parameters(
            update_global_from_local(weights_results, self.learning_rate)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif rnd == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate client evaluation results one at a time, asynchronously."""
        return super().aggregate_evaluate(rnd, results, failures)()
