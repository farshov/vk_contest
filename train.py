import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import numpy as np

from tqdm import tqdm
import time
from utils import sigmoid
import sys
import warnings


def train_classifier(model, embedder, train, test, loss_fn, output_dim,
          lr=0.001, batch_size=1024, n_epochs=4, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    train_loader = torch_data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch_data.DataLoader(test, batch_size=batch_size, shuffle=False)

    model = model.cuda()
    for epoch in range(n_epochs):
        print('epoch:', epoch)
        start_time = time.time()

        scheduler.step()

        model.train()
        avg_loss = 0.
        i = 0
        print_every = 5
        for x_batch, y_batch, lengths in tqdm(train_loader, disable=False):

            x_batch = torch.tensor(embedder.wv[np.array(x_batch).flatten()].reshape((-1, 200,
                                                                                     model.emb_size))).to(device)
            if not sys.warnoptions:
                warnings.simplefilter("ignore")
                y_pred = model(x_batch, lengths).to(device)
            loss = loss_fn(y_pred, y_batch.to(y_pred)).to(device)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            if not i % print_every:
                print(f'{i}th Loss:', '%.4f' % loss.item())
            i += 1
        model.eval()
        test_preds = np.zeros((len(test), output_dim))

        for i, x_batch, lengths in enumerate(test_loader):
            x_batch = torch.tensor(embedder.wv[np.array(x_batch).flatten()].reshape((-1, 200,
                                                                                     model.emb_size)))
            y_pred = sigmoid(model(x_batch, lengths).detach().cpu().numpy())

            test_preds[i * batch_size:(i + 1) * batch_size, :] = y_pred

        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, elapsed_time))

    return test_preds
