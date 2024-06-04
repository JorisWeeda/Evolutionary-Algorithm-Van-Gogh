import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def gaussian(alpha):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi


class RBF(nn.Module):
    def __init__(self, in_features, out_features, basis_func=gaussian):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))

        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)


class RBFN(nn.Module):
    def __init__(self, input_dim, num_centers, output_dim):
        super(RBFN, self).__init__()

        self.rbf_layer = RBF(input_dim, num_centers)
        self.linear_layer = nn.Linear(num_centers, output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, x_a, x_b):
        x_a_T = torch.FloatTensor(x_a)
        x_b_T = torch.FloatTensor(x_b)

        u = self.normalize(2 * x_a_T - x_b_T)

        y = self.rbf_layer(u)
        y = self.linear_layer(y)
        return y

    def train_rbf(self, x_a, x_b, desired_output, epochs=1, batch_size=32):
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_a), torch.FloatTensor(x_b), torch.FloatTensor(desired_output))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            self.train()
            total_loss = 0
            for batch_x_a, batch_x_b, batch_y in dataloader:
                self.optimizer.zero_grad()

                p = self.forward(batch_x_a, batch_x_b)
                loss = self.loss_func(p, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

        return avg_loss

    @staticmethod
    def normalize(x):
        min_val = torch.min(x, dim=0, keepdim=True)[0]
        max_val = torch.max(x, dim=0, keepdim=True)[0]
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Avoid division by zero
        return -1 + 2 * (x - min_val) / range_val

# Example usage
if __name__ == "__main__":
    input_dim = 500
    num_centers = 500  # Adjust this based on your needs
    output_dim = 500

    model = RBFN(input_dim, num_centers, output_dim)

    # Create some dummy data
    x_a = np.random.rand(100, input_dim)  # 100 samples, each of dimension 500
    x_b = np.random.rand(100, input_dim)  # Another set of 100 samples, each of dimension 500
    y = np.random.randint(0, 2, (100, output_dim))  # Example: binary output with 0 or 1

    # Train the model
    model.train_rbf(x_a, x_b, y, epochs=100, batch_size=32)
