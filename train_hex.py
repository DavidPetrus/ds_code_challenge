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
flags.DEFINE_integer('num_workers',0,'')
flags.DEFINE_float('lr',1e-2,'')
flags.DEFINE_integer('num_epochs',20,'')
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
    weekly_counts = defaultdict(lambda: [0] * 51)
    startdate = datetime(2020,1,5,0,0,0)
    date_idxs = {}
    for d in range(51): date_idxs[startdate + d*timedelta(days=7)] = d

    for hex_key, reqs in hex_dict.items():
        for req in reqs:
            req_dt = datetime.strptime(req[2][:-6], format_string)
            week_offset = (req_dt - startdate).days // 7
            if week_offset < 0 or week_offset >= 51: continue
            weekly_counts[hex_key][date_idxs[startdate + week_offset*timedelta(days=7)]] += 1

    start = datetime.now()
    print(len(weekly_counts.keys()))
    training_set = TimeSeriesDataset(list(weekly_counts.keys())[:750], weekly_counts)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    validation_set = TimeSeriesDataset(list(weekly_counts.keys())[750:], weekly_counts)
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

            result = predictor(request_batch) # bs, 2
            loss = F.mse_loss(result, target_batch, reduction="mean")
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                slope_diff = (result[:,0] - target_batch).abs().mean()
                bias_diff = (result[:,1] - target_batch).abs().mean()

            if train_iter % 10 == 1:
                print(f"Epoch: {epoch}, Train Iter: {train_iter}, Loss: {loss:.2f}, Slope diff: {slope_diff:.2f}, Bias diff: {bias_diff:.2f}")

            if FLAGS.wandb:
                wandb.log({"Epoch": epoch, "Train Iter": train_iter, "Loss": loss, "Slope diff": slope_diff, "Bias diff": bias_diff})

        for item in valloader:
            with torch.no_grad():
                val_iter += 1
                request_batch, target_batch = item
                request_batch = request_batch.to("cuda") # bs,9,c,h,w
                target_batch = target_batch.to("cuda") # bs

                result = predictor(request_batch) # bs
                loss = F.mse_loss(result, target_batch, reduction="mean")

                slope_diff = (result[:,0] - target_batch).abs().mean()
                bias_diff = (result[:,1] - target_batch).abs().mean()

                if val_iter % 10 == 1:
                    print(f"Val Iter: {val_iter}, Loss: {loss:.2f}, Slope diff: {slope_diff:.2f}, Bias diff: {bias_diff:.2f}")

                if FLAGS.wandb:
                    wandb.log({"Epoch": epoch, "Val Iter": val_iter, "Val loss": loss, "Val Slope diff": slope_diff, "Val Bias diff": bias_diff})

        if epoch % 1 == 0:
            torch.save(predictor.state_dict(),f"weights/{FLAGS.exp}_{epoch}.pt")


if __name__ == '__main__':
    app.run(main)