import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def gaussian(alpha):
    """
    Applies a Gaussian radial basis function to the input tensor.

    Args:
        alpha (torch.Tensor): The input tensor for the Gaussian function.

    Returns:
        torch.Tensor: The result of applying the Gaussian function.
    """
    return torch.exp(-1 * alpha.pow(2))


class RBF(nn.Module):
    """
    Radial Basis Function (RBF) layer for neural networks.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features (number of RBF centers).
        basis_func (callable): The basis function to use (default: Gaussian).
    """

    def __init__(self, in_features, out_features, basis_func=gaussian):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis_func = basis_func

        # Parameters for the RBF centers and their scales (log_sigmas)
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the RBF centers and log_sigmas parameters.
        """
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        """
        Forward pass through the RBF layer.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: The output tensor after applying the RBF transformation.
        """
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)


class RBFN(nn.Module):
    """
    Radial Basis Function Network (RBFN) that includes an RBF layer and a linear layer for classification or regression.

    Args:
        input_dim (int): The dimension of the input features.
        num_centers (int): The number of RBF centers (hidden units in the RBF layer).
        output_dim (int): The dimension of the output features.
    """

    def __init__(self, input_dim, num_centers, output_dim):
        super(RBFN, self).__init__()

        # Define the RBF layer and linear output layer
        self.rbf_layer = RBF(input_dim, num_centers)
        self.linear_layer = nn.Linear(num_centers, output_dim)
        self.sigmoid = nn.Sigmoid()

        # Optimizer and loss function for training
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_func = nn.BCELoss()

    def forward(self, x_a, x_b):
        """
        Forward pass through the RBFN.

        Args:
            x_a (np.ndarray): First input array of shape (batch_size, input_dim).
            x_b (np.ndarray): Second input array of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x_a_T = torch.FloatTensor(x_a)
        x_b_T = torch.FloatTensor(x_b)

        # Normalize the inputs
        u = self._normalize(2 * x_a_T - x_b_T)

        # Pass through the RBF and linear layers
        y = self.rbf_layer(u)
        y = self.linear_layer(y)
        y = self.sigmoid(y)
        return y

    def train_rbf(self, x_a, x_b, desired_output, epochs=10, batch_size=32):
        """
        Trains the RBF network on the provided data.

        Args:
            x_a (np.ndarray): First input array of shape (num_samples, input_dim).
            x_b (np.ndarray): Second input array of shape (num_samples, input_dim).
            desired_output (np.ndarray): Desired output array of shape (num_samples, output_dim).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            float: The average loss over the final epoch.
        """
        # Create a PyTorch dataset and dataloader
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_a), torch.FloatTensor(x_b), torch.FloatTensor(desired_output))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for _ in range(epochs):
            self.train()
            total_loss = 0

            for batch_x_a, batch_x_b, batch_y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.forward(batch_x_a, batch_x_b)

                loss = self.loss_func(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

        return avg_loss

    @staticmethod
    def _normalize(x):
        """
        Normalizes the input tensor to the range [-1, 1].

        Args:
            x (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        min_val = torch.min(x, dim=0, keepdim=True)[0]
        max_val = torch.max(x, dim=0, keepdim=True)[0]
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Prevent division by zero
        return -1 + 2 * (x - min_val) / range_val


if __name__ == "__main__":
    # Example usage of the RBFN model
    input_dim = 500
    num_centers = 500
    output_dim = 500

    model = RBFN(input_dim, num_centers, output_dim)

    # Generate random training data
    x_a = np.random.rand(100, input_dim)
    x_b = np.random.rand(100, input_dim)
    y = np.random.randint(0, 2, (100, output_dim))

    # Train the model
    model.train_rbf(x_a, x_b, y, epochs=100, batch_size=32)
