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
flags.DEFINE_integer('num_epochs',1000,'')


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

    weekly_counts = defaultdict(lambda: defaultdict(lambda: 0))
    startdate = datetime(2020,1,5,0,0,0)
    for hex_key, reqs in hex_dict.items():
        for req in reqs:
            req_dt = datetime.strptime(req[2][:-6], format_string)
            week_offset = (req_dt - startdate).days // 7
            weekly_counts[hex_key][startdate + week_offset*timedelta(days=7)] += 1

    start = datetime.now()
    training_set = TimeSeriesDataset(list(hex_dict.keys())[:1500], weekly_counts)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    validation_set = TimeSeriesDataset(list(hex_dict.keys())[1500:], weekly_counts)
    valloader = torch.utils.data.DataLoader(validation_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    predictor = RequestPredictor()
    #predictor.to("cuda")

    optimizer = torch.optim.Adam(params=predictor.parameters(), lr=FLAGS.lr)
    train_iter, val_iter = 0, 0

    for epoch in range(FLAGS.num_epochs):
        for item in trainloader:
            optimizer.zero_grad()

            train_iter += 1
            request_batch, target_batch = item
            request_batch = request_batch#.to("cuda")
            target_batch = target_batch#.to("cuda")

            result = predictor(request_batch) # bs
            loss = F.mse_loss(result, target_batch, reduction="mean")
            loss.backward()

            optimizer.step()

            if train_iter % 100 == 1:
                print(f"Epoch: {epoch}, Train Iter: {train_iter}, Loss: {loss:.2f}, Det Pools: {det_pools:.2f}, Miss Pools: {miss_pools:.2f}, False Pos: {false_pools:.2f}")

            if FLAGS.wandb:
                wandb.log({"Epoch": epoch, "Train Iter": train_iter, "Loss": loss})

        for item in valloader:
            with torch.no_grad():
                val_iter += 1
                request_batch, target_batch = item
                request_batch = request_batch#.to("cuda") # bs,9,c,h,w
                target_batch = target_batch#.to("cuda") # bs

                result = predictor(request_batch) # bs
                loss = F.mse_loss(result, target_batch, reduction="mean")

                if val_iter % 100 == 1:
                    print(f"Val Iter: {val_iter}, Loss: {loss:.2f}")

                if FLAGS.wandb:
                    wandb.log({"Epoch": epoch, "Val Iter": val_iter, "Val loss": loss})

        torch.save(predictor.state_dict(),f"weights/{FLAGS.exp}_{epoch}.pt")


if __name__ == '__main__':
    app.run(main)