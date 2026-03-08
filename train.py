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
flags.DEFINE_integer('batch_size',64,'')
flags.DEFINE_integer('num_workers',0,'')
flags.DEFINE_float('lr',1e-4,'')
flags.DEFINE_integer('num_epochs',100,'')


def main(argv):
    start = datetime.datetime.now()
    training_set = SwimmingPoolDataset("data")
    dataloader = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    classifier = PoolClassifier()

    optimizer = torch.optim.Adam(params=classifier.parameters(), lr=FLAGS.lr)

    for epoch in range(FLAGS.num_epochs):
        for item in dataloader:
            image_batch, target_batch = item
            image_batch = image_batch.to("cuda")
            target_batch = target_batch.to("cuda")

            result = classifier(image_batch) # bs, 4

            loss = F.binary_cross_entropy_with_logits(result, target_batch)


            loss.backward()

            optimizer.step()



if __name__ == '__main__':
    app.run(main)