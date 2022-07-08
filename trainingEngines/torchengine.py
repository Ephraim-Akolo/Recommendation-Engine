import torch
import numpy as np


class RecommendationEngine(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, add_user_bias=False, add_product_bias=False):
        super().__init__()
        self.add_user_bias = add_user_bias
        self.add_product_bias = add_product_bias
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.user_factors.weight.data.uniform_(0, 1)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.item_factors.weight.data.uniform_(0, 1)
        self.user_biases = torch.nn.Embedding(n_users, 1)
        self.user_biases.weight.data.uniform_(0, 0.01)
        self.item_biases = torch.nn.Embedding(n_items, 1)
        self.item_biases.weight.data.uniform_(0, 0.01)
    
    def forward(self, user, item):
        if self.add_product_bias and self.add_product_bias:
            return self._forward4(user, item)
        elif self.add_product_bias:
            return self._forward3(user, item)
        elif self.add_user_bias:
            return self._forward2(user, item)
        else:
            return self._forward1(user, item)
    
    def _forward1(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)
    
    def _forward2(self, user, item):
        pred = self.user_biases(user)
        pred += (
            (self.user_factors(user) * self.item_factors(item))
            .sum(dim=1, keepdim=True)
        )
        return pred.squeeze()

    def _forward3(self, user, item):
        pred = self.item_biases(item)
        pred += (
            (self.user_factors(user) * self.item_factors(item))
            .sum(dim=1, keepdim=True)
        )
        return pred.squeeze()

    def _forward4(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (
            (self.user_factors(user) * self.item_factors(item))
            .sum(dim=1, keepdim=True)
        )
        return pred.squeeze()
    
    def train(self, ratings:np.ndarray, epoch=1000, lr=1e-2, l2=1e-3):
        # Sort our data
        rows, cols = ratings.nonzero()
        p = np.random.permutation(len(rows))
        rows, cols = rows[p], cols[p]
        # initialize cost function
        loss_func = torch.nn.MSELoss()
        # initialize optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=l2)
        for _ in range(epoch):
            for row, col in zip(*(rows, cols)):
                # Set gradients to zero
                optimizer.zero_grad()
                # Turn data into tensors
                rating = torch.FloatTensor([ratings[row, col]])
                row = torch.LongTensor([row])
                col = torch.LongTensor([col])

                # Predict and calculate loss
                #prediction = model(row, col)
                prediction = self.forward(row, col)
                loss = loss_func(prediction, rating)
                # Backpropagate
                loss.backward()

                # Update the parameters
                optimizer.step()
            print(loss)
        print(self.user_factors)
