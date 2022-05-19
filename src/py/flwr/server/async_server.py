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

    def fit_round(
        self, rnd: int
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        raise NotImplementedError("TODO")

    def fit(self, num_rounds: int) -> History:
        """Run async FL for a number of rounds."""
        # In async, instead of each round being a fixed sample of devices,
        # each round is emptying the buffer and aggregating client results from
        # it once. Suggested buffer size k = 10.
        raise NotImplementedError("TODO")

    def fit_clients_async(self):
        """Refine parameters asynchronously on selected clients."""
        # TODO Create a thread pool and submit an executor for each
        # client_instructions But instead of waiting for all results, buffer the
        # results in a multiprocessing Queue or Deque and pop k results from
        # the buffer once k results are available in it.
