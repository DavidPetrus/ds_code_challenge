import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import glob
import os

from absl import flags

FLAGS = flags.FLAGS


class SwimmingPoolDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):

        print(f"{data_dir}/yes/*.txt")
        label_files = glob.glob(f"{data_dir}/yes/*.txt")
        self.labels = []
        self.images = []
        for label_file in label_files:
            with open(label_file, "r") as fp: annots = fp.read().splitlines()
            img_bboxes = []
            for annot in annots:
                splits = annot.split(" ")
                img_bboxes.append((max(0, int(splits[0])), max(0, int(splits[1])), min(int(splits[2]), 1250), min(int(splits[3]), 1250), splits[4]))

            self.labels.append(img_bboxes)
            self.images.append(label_file.replace(".txt", ".tif"))

        self.images.extend(glob.glob(f"{data_dir}/no/*.tif"))
        self.image_size = 417
        print(len(self.labels), len(self.images))

    def crop_pool_image(self, bbox):
        lx = min(np.random.randint(max(0, bbox[2]-self.image_size), bbox[0]+1), 1250-self.image_size)
        ty = min(np.random.randint(max(0, bbox[3]-self.image_size), bbox[1]+1), 1250-self.image_size)

        return (lx, ty, lx+self.image_size, ty+self.image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]))
        
        label = self.labels[index] if index < len(self.labels) else None
        if label == []:
            image = np.array(Image.open(self.images[np.random.randint(len(self.labels), len(self.images))]))
            label = None

        if label is not None:
            # Contains pool
            label = label[np.random.randint(len(label))]
            crop_coords = self.crop_pool_image(label)
            crop_image = image[crop_coords[0]:crop_coords[2], crop_coords[1]:crop_coords[3]]
            pool = 1.
            if crop_image.shape != (417, 417, 3):
                print(image.shape, crop_image.shape, crop_coords)
        else:
            # No pool
            lx, ty = np.random.randint(0, 1250-self.image_size), np.random.randint(0, 1250-self.image_size)
            crop_image = image[lx: lx+self.image_size, ty: ty+self.image_size]
            pool = 0.
            if crop_image.shape != (417, 417, 3):
                print(image.shape, crop_image.shape, lx, ty)


        img = torch.from_numpy(np.ascontiguousarray(crop_image)).float()
        img = img / 255
        img = img.movedim(2,0)

        return img, pool


class TimeSeries(torch.utils.data.Dataset):
    def __init__(self, hex_keys, weekly_counts):
        self.seq_len = 26
        self.hex_keys = hex_keys
        self.weekly_counts = weekly_counts # num_hexs, num_weeks

        self.add_neighbors = False

    def __len__(self):
        return len(self.weekly_counts)

    def __getitem__(self, index):
        seq_idx = np.random.randint(51 - (self.seq_len + 4))
        input_seq = self.weekly_counts[index][seq_idx: seq_idx+self.seq_len]
        target_seq = self.weekly_counts[index][seq_idx+self.seq_len: seq_idx+self.seq_len+4]
        if self.add_neighbors:
            neighbor_seqs = []
            for hex_key in h3.k_ring(self.hex_keys[index]):
                n_idx = self.hex_keys.index(hex_key)
                neighbor_seqs.append(self.weekly_counts[n_idx][seq_idx: seq_idx+self.seq_len])
                input_seq.extend(neighbor_seqs[-1])
                    
        return input_seq, target_seq