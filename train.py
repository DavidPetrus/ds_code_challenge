import numpy as np
import cv2
import torch
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
flags.DEFINE_string('data_dir','C:\\Users\\david\\ds_code_challenge\\swimming_pools','')
flags.DEFINE_integer('batch_size',64,'')
flags.DEFINE_integer('num_workers',0,'')
flags.DEFINE_float('lr',1e-4,'')
flags.DEFINE_integer('num_epochs',100,'')


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

    optimizer = torch.optim.Adam(params=classifier.parameters(), lr=FLAGS.lr)
    train_iter = 0

    for epoch in range(FLAGS.num_epochs):
        for item in dataloader:
            train_iter += 1
            image_batch, target_batch = item
            image_batch = image_batch#.to("cuda")
            target_batch = target_batch#.to("cuda")

            result = classifier(image_batch) # bs

            loss = F.binary_cross_entropy_with_logits(result, target_batch)

            loss.backward()

            with torch.no_grad():
                num_pools = target_batch.sum()
                if num_pools > 0:
                    det_pools = (result * target_batch).sum() / num_pools
                    miss_pools = ((1-result) * target_batch).sum() / num_pools
                    false_pools = (result * (1-target_batch)).sum() / result.sum()
                else:
                    det_pools, miss_pools, false_pools = 0, 0, 0

            optimizer.step()

            if FLAGS.wandb:
                wandb.log({"Epoch": epoch, "Train Iter": train_iter, "Loss": loss, "Detected Pools": det_pools, "Missed Pools": miss_pools, \
                    "False Positives": false_pools})


if __name__ == '__main__':
    app.run(main)