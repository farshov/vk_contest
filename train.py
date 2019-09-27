import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from collections import OrderedDict
from tqdm import tqdm

from utils import CommentsDataset


def train(sent_train, targets_train, len_train, sent_dev, targets_dev, len_dev, model,
          device, batch_size=64, random_state=42, num_epochs=100):

    step = 0
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters())
    lr_politic = lambda epoch: epoch * 0.8
    scheduler = sched.LambdaLR(optimizer, lr_politic)  # Constant LR

    train_dataset = CommentsDataset(sent_train, targets_train, len_train)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4)
    dev_dataset = CommentsDataset(sent_dev, targets_dev, len_dev)

    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4)

    steps_till_eval = len(train_dataset) // 100
    epoch = step // len(train_dataset)
    while epoch != num_epochs:
        epoch += 1
        print(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:

            for X, y, lenghts in train_loader:

                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                output = model(X, lenghts)
                loss = torch.tensor(0).to(output)
                for i in range(output.shape[1]):
                    cur_out = output[:, i].view(-1, 1)
                    cur_out = torch.cat((torch.zeros_like(cur_out)-cur_out, cur_out), dim=1)
                    loss += F.nll_loss(cur_out, y[:, i])
                loss_val = loss.item()

                # Backward
                loss.backward()
                optimizer.step()
                scheduler.step(step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                # if not step % 2048:
                progress_bar.set_postfix(epoch=epoch,
                                          NLL=loss_val)

                steps_till_eval -= batch_size
                # if steps_till_eval <= 0:
                #     steps_till_eval = len(train_dataset) // 100
                #
                #     results, pred_dict = evaluate(model, dev_loader, device, batch_size)
                #
                #     # Log to console
                #     results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
    return model


def evaluate(model, data_loader, device, batch_size):

    model.eval()
    pred_dict = {}

    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            output = model(X)
            loss = torch.tensor(0)
            for i in range(output.shape[1]):
                cur_out = output[:, i]
                cur_out = torch.cat((torch.zeros_like(cur_out) - cur_out, cur_out), dim=1)
                loss += F.nll_loss(cur_out, y[:, i])


            # # Get F1 and EM scores
            # p1, p2 = log_p1.exp(), log_p2.exp()
            # starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)
            #
            # # Log info
            # progress_bar.update(batch_size)
            # progress_bar.set_postfix(NLL=nll_meter.avg)
            #
            # preds, _ = util.convert_tokens(gold_dict,
            #                                ids.tolist(),
            #                                starts.tolist(),
            #                                ends.tolist(),
            #                                use_squad_v2)
            # pred_dict.update(preds)

    model.train()

    # results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    # if use_squad_v2:
    #     results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict
