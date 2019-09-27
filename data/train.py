import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from tqdm import tqdm
from ujson import load as json_load

from utils import CommentsDataset


def train(sent_train, targets_train, sent_dev, targets_dev, model,
          device, batch_size=64, random_state=42, num_epochs=100):

    word_vectors = util.torch_from_json(args.word_emb_file)

    step = 0
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters())
    lr_politic = lambda epoch: epoch * 0.8
    scheduler = sched.LambdaLR(optimizer, lr_politic)  # Constant LR

    ema = util.EMA(model, args.ema_decay)
    train_dataset = CommentsDataset(sent_train, targets_train)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=-1)
    dev_dataset = CommentsDataset(sent_dev, targets_dev)

    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=-1)

    steps_till_eval = len(train_dataset) // 10
    epoch = step // len(train_dataset)
    while epoch != num_epochs:
        epoch += 1
        print(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:

            for X, y in train_loader:

                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                output = model(X)
                loss = 0
                for i in range(output.shape[1]):
                    cur_out = output[:, i]
                    cur_out = torch.cat((torch.zeros_like(cur_out)-cur_out, cur_out), dim=1)
                    loss += F.nll_loss(cur_out, y[:, i])
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            # log_p1, log_p2 = model(cw_idxs, qw_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict
