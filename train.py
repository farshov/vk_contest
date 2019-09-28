import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.DataLoader as DataLoader
import numpy as np

from tqdm import tqdm
import time
from utils import sigmoid


def train(model, train, test, loss_fn, output_dim,
          lr=0.001, batch_size=512, n_epochs=4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):
        print('epoch:', epoch)
        start_time = time.time()

        scheduler.step()

        model.train()
        avg_loss = 0.

        for data in tqdm(train_loader, disable=False):
            x_batch = data[:-1]
            y_batch = data[-1]

            y_pred = model(*x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        test_preds = np.zeros((len(test), output_dim))

        for i, x_batch in enumerate(test_loader):
            y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())

            test_preds[i * batch_size:(i + 1) * batch_size, :] = y_pred

        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, elapsed_time))

    return test_preds
