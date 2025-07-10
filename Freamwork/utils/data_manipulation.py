import torch

  # Split data into training and validation sets (70/30 split)
def split_data(x,y,test_size):
        indices = torch.randperm(len(x))
        size = int(len(x) *test_size)
        x = x[indices]
        y = y[indices]
        return x[:size], y[:size], x[size:], y[size:]

    # Perform k-fold cross-validation split
def cross_validation(self, k):
        fold_len = len(self.x) // k
        indices = torch.randperm(len(self.x))
        x = self.x[indices]
        y = self.y[indices]
        x_folds, y_folds = [], []
        for i in range(k):
            x_folds.append(x[fold_len * i: fold_len * (i + 1)])
            y_folds.append(y[fold_len * i: fold_len * (i + 1)])
        return x_folds, y_folds