"""Differentially Private Federated Averaging Strategy.

Author: Pooja Nadagouda

This class is adapted from:
https://github.com/matturche/flower_opacus_example/blob/main/flower_helpers.py 
"""

from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

import flwr.server.server as server
from flwr.common import FitRes, Parameters, Scalar, Weights
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgDp(FedAvg):
    """This class implements the FedAvg strategy for a Differential Privacy context.

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 0.1.
    fraction_eval : float, optional
        Fraction of clients used during validation. Defaults to 0.1.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_eval_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    """

    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        FedAvg.__init__(
            self,
            fraction_fit,
            fraction_eval,
            min_fit_clients,
            min_eval_clients,
            min_available_clients,
            eval_fn,
            on_fit_config_fn,
            on_evaluate_config_fn,
            accept_failures,
            initial_parameters,
        )

        # This variable is used to track maximum possible value of ε - privacy budget
        self.max_epsilon = 0.0

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate training results.

        Parameters
        ----------
        rnd : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[BaseException]
            Exceptions that occurred while the server was waiting for client
            updates.

        Returns
        -------
        parameters: Parameters (optional)
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        # If results is empty then return None
        if not results:
            return None

        # From the results fetch the value of ε - privacy budget for each client.
        accepted_results = []
        epsilon_values = []
        for client, result in results:
            # Check each client's accept flag so we can filter out clients who exceeded max ε.
            if result.metrics["accept"]:
                accepted_results.append([client, result])
                epsilon_values.append(result.metrics["epsilon"])
            else:
                # If any of the client's privacy budget (ε) is exceeded it is disconnected.
                server.reconnect_client(client, server.Reconnect(None))

        # Now the results contain only results of client's whose ε is within boundary
        results = accepted_results
        # Log the highest ε value among all accepted clients for the user's info.
        # This indicates how much of the privacy budget remains for subsequent training.
        if epsilon_values:
            self.max_epsilon = max(self.max_epsilon, max(epsilon_values))
        logger.info("Privacy budget ε at round {}: {:.3f}", rnd, self.max_epsilon)

        # Invoke aggregate fit logic of the parent class (FedAvg) on selected clients' results
        return super().aggregate_fit(rnd, results, failures)
