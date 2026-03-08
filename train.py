import numpy as np
import cv2
import torch
import torch.nn.functional as F
import glob
import datetime

from model import PoolClassifier
from dataloader import SwimmingPoolDataset

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_bool('wandb', True,'')
flags.DEFINE_string('data_dir','/home/usergpu/data','')
flags.DEFINE_string('weights','','')
flags.DEFINE_integer('batch_size',64,'')
flags.DEFINE_integer('num_workers',64,'')
flags.DEFINE_float('lr',1e-4,'')
flags.DEFINE_integer('num_epochs',2000,'')
flags.DEFINE_float('pool_thresh',0.5,'')


def main(argv):
    if FLAGS.wandb:
        wandb.init(project="pool_classification",name=FLAGS.exp)
        wandb.save("train.py")
        wandb.save("model.py")
        wandb.save("dataloader.py")
        wandb.config.update(flags.FLAGS)

    start = datetime.datetime.now()
    training_set = SwimmingPoolDataset(FLAGS.data_dir, val=False)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    validation_set = SwimmingPoolDataset(FLAGS.data_dir, val=True)
    valloader = torch.utils.data.DataLoader(validation_set, batch_size=int(FLAGS.batch_size / 9), shuffle=True, num_workers=FLAGS.num_workers)

    classifier = PoolClassifier()
    if FLAGS.weights: classifier.load_state_dict(torch.load(FLAGS.weights, weights_only=True))
    classifier.to("cuda")

    optimizer = torch.optim.Adam(params=classifier.parameters(), lr=FLAGS.lr)
    train_iter, val_iter = 0, 0

    for epoch in range(FLAGS.num_epochs):
        for item in trainloader:
            optimizer.zero_grad()

            train_iter += 1
            image_batch, target_batch = item
            image_batch = image_batch.to("cuda")
            target_batch = target_batch.to("cuda")

            result = classifier(image_batch) # bs
            loss = F.binary_cross_entropy_with_logits(result, target_batch, reduction="sum")

            loss.backward()

            with torch.no_grad():
                pool_infers = torch.where(result.sigmoid() > FLAGS.pool_thresh, 1., 0.)
                num_pools = target_batch.sum()
                if num_pools > 0:
                    det_pools = (pool_infers * target_batch).sum() / num_pools
                    miss_pools = ((1-pool_infers) * target_batch).sum() / num_pools
                    false_pools = (pool_infers * (1-target_batch)).sum() / pool_infers.sum()
                else:
                    det_pools, miss_pools, false_pools = 0, 0, 0

            optimizer.step()

            if train_iter % 100 == 1:
                print(f"Epoch: {epoch}, Train Iter: {train_iter}, Loss: {loss:.2f}, Det Pools: {det_pools:.2f}, Miss Pools: {miss_pools:.2f}, False Pos: {false_pools:.2f}")

            if FLAGS.wandb:
                wandb.log({"Epoch": epoch, "Train Iter": train_iter, "Loss": loss, "Detected Pools": det_pools, "Missed Pools": miss_pools, \
                    "False Positives": false_pools})

        if epoch % 5 == 0:
            for item in valloader:
                with torch.no_grad():
                    val_iter += 1
                    image_batch, target_batch = item
                    image_batch = image_batch.to("cuda") # bs,9,c,h,w
                    target_batch = target_batch.to("cuda") # bs

                    image_batch = image_batch.reshape(-1, 3, 417, 417)
                    result = classifier(image_batch).reshape(-1, 9) # bs,9
                    pool_infers = torch.where(result.sigmoid() > FLAGS.pool_thresh, 1., 0.)
                    pool_infers = pool_infers.max(dim=1)[0] # bs

                    num_pools = target_batch.sum()
                    if num_pools > 0:
                        det_pools = (pool_infers * target_batch).sum() / num_pools
                        miss_pools = ((1-pool_infers) * target_batch).sum() / num_pools
                        false_pools = (pool_infers * (1-target_batch)).sum() / pool_infers.sum()
                    else:
                        det_pools, miss_pools, false_pools = 0, 0, 0

                    if val_iter % 100 == 1:
                        print(f"Val Iter: {val_iter}, Loss: {loss:.2f}, Det Pools: {det_pools:.2f}, Miss Pools: {miss_pools:.2f}, False Pos: {false_pools:.2f}")

                    if FLAGS.wandb:
                        wandb.log({"Epoch": epoch, "Val Iter": val_iter, "Val Detected Pools": det_pools, "Val Missed Pools": miss_pools, \
                            "Val False Positives": false_pools})

            torch.save(classifier.state_dict(),f"weights/{FLAGS.exp}_{epoch}.pt")



if __name__ == '__main__':
    app.run(main)