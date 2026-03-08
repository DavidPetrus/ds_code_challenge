import numpy as np
import cv2
import torch
import torch.nn.functional as F
import glob
import datetime
import random

from model import PoolClassifier
from dataloader import SwimmingPoolDataset

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_bool('wandb', True,'')
flags.DEFINE_string('data_dir','/home/usergpu/data','')
flags.DEFINE_integer('batch_size',128,'')
flags.DEFINE_integer('num_workers',64,'')
flags.DEFINE_float('lr',1e-4,'')
flags.DEFINE_integer('num_epochs',1000,'')
flags.DEFINE_float('pool_thresh',0.5,'')


def main(argv):
    if FLAGS.wandb:
        wandb.init(project="pool_classification",name=FLAGS.exp)
        wandb.save("train.py")
        wandb.save("model.py")
        wandb.save("dataloader.py")
        wandb.config.update(flags.FLAGS)

    start = datetime.datetime.now()
    training_set = SwimmingPoolDataset(FLAGS.data_dir)
    dataloader = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    classifier = PoolClassifier()
    classifier.to("cuda")

    optimizer = torch.optim.Adam(params=classifier.parameters(), lr=FLAGS.lr)
    train_iter = 0

    for epoch in range(FLAGS.num_epochs):
        for item in dataloader:
            optimizer.zero_grad()

            train_iter += 1
            image_batch, target_batch = item
            image_batch = image_batch.to("cuda")
            target_batch = target_batch.to("cuda")

            result = classifier(image_batch) # bs
            loss = F.binary_cross_entropy_with_logits(result, target_batch)

            loss.backward()

            with torch.no_grad():
                pool_infers = torch.where(result > FLAGS.pool_thresh, 1., 0.)
                num_pools = target_batch.sum()
                if num_pools > 0:
                    det_pools = (pool_infers * target_batch).sum() / num_pools
                    miss_pools = ((1-pool_infers) * target_batch).sum() / num_pools
                    false_pools = (pool_infers * (1-target_batch)).sum() / pool_infers.sum()
                else:
                    det_pools, miss_pools, false_pools = 0, 0, 0

            optimizer.step()

            if train_iter % 10 == 1:
                print(f"Train Iter: {train_iter}, Loss: {loss}, Det Pools: {det_pools}, Miss Pools: {miss_pools}, False Pos: {false_pools}")

            if FLAGS.wandb:
                wandb.log({"Epoch": epoch, "Train Iter": train_iter, "Loss": loss, "Detected Pools": det_pools, "Missed Pools": miss_pools, \
                    "False Positives": false_pools})


if __name__ == '__main__':
    app.run(main)