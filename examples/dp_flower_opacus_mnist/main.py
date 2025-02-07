import sys
import flwr as fl
import multiprocessing as mp
import argparse
sys.path.insert(0,"..")
from src.py.flwr.server.strategy.fedAvgDP import test
from src.py.flwr.client.dp_client import DPClient
from src.py.flwr.server.strategy.fedAvgDP import FedAvgDp
from opacus import PrivacyEngine
import torch
import torch.nn as nn
from torch.nn import BCELoss, Linear, Module
from tensorflow.keras.datasets import mnist
from typing import List, Tuple
import numpy as np


# Training Model that is being used by the client
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.gn1 = nn.GroupNorm(int(6 / 3), 6)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(int(16 / 4), 16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.gn1(x)
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.gn2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
#function to load the MNIST dataset
XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]
def load(
    num_partitions: int,
) -> PartitionedDataset:
    """Create partitioned version of MNIST."""
    xy_train, xy_test = tf.keras.datasets.mnist.load_data()
    xy_train_partitions = create_partitions(xy_train, num_partitions)
    xy_test_partitions = create_partitions(xy_test, num_partitions)
    return list(zip(xy_train_partitions, xy_test_partitions))

# Function to get the weights of the model
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


# Main function to create DP client
def main(args) -> None:
    # Load model
    model = Net()
    
    # Load a subset of MNIST to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load(args.num_clients)[args.partition]

    # drop samples to form exact batches for dpsgd
    # this is necessary since dpsgd is sensitive to uneven batches
    # due to microbatching
    if args.dpsgd and x_train.shape[0] % args.batch_size != 0:
        drop_num = x_train.shape[0] % args.batch_size
        x_train = x_train[:-drop_num]
        y_train = y_train[:-drop_num]


    # Start Flower client
    # client = DPClient(model, x_train, y_train, x_test, y_test, args)
    
    
    optimizer = SGD(module.parameters(), lr=0.001)
    criterion = BCELoss()
    target_epsilon = 1.0
    target_delta = 0.1
    privacy_engine = PrivacyEngine()
    train_loader, test_loader = get_dataloaders(batch_size)
    
    client = DPClient(
        model,
        optimizer,
        criterion,
        privacy_engine,
        train_loader,
        test_loader,
        target_epsilon,
        target_delta,
        epochs=10,
        max_grad_norm=1.0,
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
    if args.dpsgd:
        print("Privacy Loss: ", PRIVACY_LOSS)
        
        

def get_dataloaders(batch_size):
    """Get dataloaders for unit testing."""
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X = torch.from_numpy(numpy.array(X_train))
    y = torch.from_numpy(numpy.array(Y_train))
    train_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    return train_loader, test_loader

def get_eval_fn():
    """Get the evaluation function for server side.
    
    Returns
    -------
    evaluate
        The evaluation function
    """

    def evaluate(weights):
        """Evaluation function for server side.

        Parameters
        ----------
        weights
            Updated model weights to evaluate.

        Returns
        -------
        loss
            Loss on the test set.
        accuracy
            Accuracy on the test set.
        """
        global batch_size
        # Prepare multiprocess
        manager = mp.Manager()
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process
        # 0 is for the share and 1 total number of clients, here for server test
        # we take the full test set
        p = mp.Process(target=test, args=(weights, return_dict, 0, 1, batch_size))
        # Start the process
        p.start()
        # Wait for it to end
        p.join()
        # Close it
        try:
            p.close()
        except ValueError as e:
            print(f"Couldn't close the evaluating process: {e}")
        # Get the return values
        loss = return_dict["loss"]
        accuracy = return_dict["accuracy"]
        # Del everything related to multiprocessing
        del (manager, return_dict, p)
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--num-clients",
        default=2,
        type=int,
        help="Total number of fl participants, requied to get correct partition",
    )
    parser.add_argument(
        "--partition",
        type=int,
        required=True,
        help="Data Partion to train on. Must be less than number of clients",
    )
    parser.add_argument(
        "--local-epochs",
        default=1,
        type=int,
        help="Total number of local epochs to train",
    )
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--learning-rate", default=0.15, type=float, help="Learning rate for training"
    )
    # DPSGD specific arguments
    parser.add_argument(
        "--dpsgd",
        default=False,
        type=bool,
        help="If True, train with DP-SGD. If False, " "train with vanilla SGD.",
    )
    parser.add_argument("--l2-norm-clip", default=1.0, type=float, help="Clipping norm")
    parser.add_argument(
        "--noise-multiplier",
        default=1.1,
        type=float,
        help="Ratio of the standard deviation to the clipping norm",
    )
    parser.add_argument(
        "--microbatches",
        default=32,
        type=int,
        help="Number of microbatches " "(must evenly divide batch_size)",
    )
    args = parser.parse_args()

    main(args)
    
    #----------------------------------------------------------------
    
    # NOW THAT WE HAVE DONE CLIENT LETS START THE SERVER
    
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-r", type=int, default=3, help="Number of rounds for the federated training"
    )
    
    parser.add_argument(
        "-nbc",
        type=int,
        default=2,
        help="Number of clients to keep track of dataset share",
    )
    
    parser.add_argument("-b", type=int, default=256, help="Batch size")
    
    parser.add_argument(
        "-fc",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    
    parser.add_argument(
        "-ac",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    
    args = parser.parse_args()
    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)
    global batch_size
    batch_size = int(args.b)
    
    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")
    # Create a new fresh model to initialize parameters
    net = Net()
    init_weights = get_weights(net)
    # Convert the weights (np.ndarray) to parameters
    init_param = fl.common.weights_to_parameters(init_weights)
    
    # del the net as we don't need it anymore
    del net
    
    # Define the strategy
    strategy = FedAvgDp(
        fraction_fit=float(fc / ac),
        min_fit_clients=fc,
        min_available_clients=ac,
        eval_fn=get_eval_fn(),
        initial_parameters=init_param,
    )
    fl.server.start_server(
        "[::]:8080", config={"num_rounds": rounds}, strategy=strategy
    )
    
