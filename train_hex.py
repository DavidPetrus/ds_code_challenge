import numpy as np
import cv2
import torch
import torch.nn.functional as F
import glob
from datetime import datetime, timedelta
import csv
from collections import defaultdict

from model import RequestPredictor
from dataloader import TimeSeriesDataset

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_bool('wandb', True,'')
flags.DEFINE_integer('batch_size',64,'')
flags.DEFINE_integer('num_workers',64,'')
flags.DEFINE_float('lr',1e-2,'')
flags.DEFINE_integer('num_epochs',100,'')
flags.DEFINE_float('l1_beta',10.,'')


def main(argv):
    if FLAGS.wandb:
        wandb.init(project="request_prediction",name=FLAGS.exp)
        wandb.save("train_hex.py")
        wandb.save("model.py")
        wandb.save("dataloader.py")
        wandb.config.update(flags.FLAGS)

    filename = "sr_hex.csv"
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        data_read = [row for row in reader]

    hex_dict = defaultdict(list)
    for req in data_read[1:]:
        hex_dict[req[-1]].append(req)

    print(data_read[0])
    print(data_read[1][2])
    format_string = "%Y-%m-%d %H:%M:%S"

    # Add zero weeks
    weekly_counts = defaultdict(lambda: defaultdict(lambda: 0))
    startdate = datetime(2020,1,5,0,0,0)
    for hex_key, reqs in hex_dict.items():
        for req in reqs:
            req_dt = datetime.strptime(req[2][:-6], format_string)
            week_offset = (req_dt - startdate).days // 7
            weekly_counts[hex_key][startdate + week_offset*timedelta(days=7)] += 1

    data_dict = {}
    for hex_key, req_counts in weekly_counts.items():
        sorted_dates = sorted(list(req_counts.keys()))
        if len(sorted_dates) < 50: continue
        data_dict[hex_key] = []
        for date in sorted_dates: data_dict[hex_key].append(req_counts[date])

    start = datetime.now()
    print(len(data_dict.keys()))
    training_set = TimeSeriesDataset(list(data_dict.keys())[:750], data_dict)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    validation_set = TimeSeriesDataset(list(data_dict.keys())[750:], data_dict)
    valloader = torch.utils.data.DataLoader(validation_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    predictor = RequestPredictor()
    predictor.to("cuda")

    optimizer = torch.optim.Adam(params=predictor.parameters(), lr=FLAGS.lr)
    train_iter, val_iter = 0, 0

    for epoch in range(FLAGS.num_epochs):
        for item in trainloader:
            optimizer.zero_grad()

            train_iter += 1
            request_batch, target_batch = item
            request_batch = request_batch.to("cuda")
            target_batch = target_batch.to("cuda")

            result = predictor(request_batch) # bs
            loss = F.smooth_l1_loss(result, target_batch, beta=FLAGS.l1_beta, reduction="mean")
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                diff = result - target_batch
                diff_frac = (diff / target_batch).abs() / target_batch

            if train_iter % 100 == 1:
                print(f"Epoch: {epoch}, Train Iter: {train_iter}, Loss: {loss:.2f}, Error_frac: {diff_frac:.2f}")

            if FLAGS.wandb:
                wandb.log({"Epoch": epoch, "Train Iter": train_iter, "Loss": loss, "Error frac": diff_frac})

        for item in valloader:
            with torch.no_grad():
                val_iter += 1
                request_batch, target_batch = item
                request_batch = request_batch.to("cuda") # bs,9,c,h,w
                target_batch = target_batch.to("cuda") # bs

                result = predictor(request_batch) # bs
                loss = F.smooth_l1_loss(result, target_batch, beta=FLAGS.l1_beta, reduction="mean")

                diff = result - target_batch
                diff_frac = (diff / target_batch).abs() / target_batch

                if val_iter % 100 == 1:
                    print(f"Val Iter: {val_iter}, Loss: {loss:.2f}, Error_frac: {diff_frac:.2f}")

                if FLAGS.wandb:
                    wandb.log({"Epoch": epoch, "Val Iter": val_iter, "Val loss": loss, "Error frac": diff_frac})

        if epoch % 10 == 0:
            torch.save(predictor.state_dict(),f"weights/{FLAGS.exp}_{epoch}.pt")


if __name__ == '__main__':
    app.run(main)