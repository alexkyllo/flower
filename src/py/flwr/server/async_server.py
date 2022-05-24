from typing import Dict, Optional, Tuple

from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import FitResultsAndFailures, Server
from flwr.server.strategy import FedAsync


class AsyncServer(Server):
    """A Flower Server that processes client results asynchronously."""

    def __init__(self, client_manager: ClientManager, strategy: Optional[FedAsync] = None) -> None:
        super().__init__(client_manager, strategy)

    def fit(self, num_rounds: int) -> History:
        """Run async FL for a number of rounds."""
        # Buffered or unbuffered async?
        #
        # In buffered async, instead of each round
        # being a fixed sample of devices, each round is emptying the buffer and
        # aggregating client results from it once. Suggested buffer size k = 10.
        #
        # In unbuffered, each round is a single client.
        #
        # TODO: Figure out how often to send global model to new clients to
        # trigger training
        raise NotImplementedError("TODO")

    def fit_round(
        self, rnd: int
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        """Fit an async "round"."""
        # call self.strategy.configure_fit()
        # call self.fit_clients_async()
        # return result (new global model parameters)
        raise NotImplementedError("TODO")

    def fit_clients_async(self):
        """Refine parameters asynchronously on selected clients."""
        # TODO Create a thread pool and submit an executor for each
        # client_instructions But instead of waiting for all results, buffer the
        # results in a multiprocessing Queue or Deque and pop k results from
        # the buffer once k results are available in it.
        raise NotImplementedError("TODO")
