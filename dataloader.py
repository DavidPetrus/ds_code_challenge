import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from PIL import Image
import cv2
import glob
import os
import h3

from absl import flags

FLAGS = flags.FLAGS


class SwimmingPoolDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, val):
        label_files = glob.glob(f"{data_dir}/yes/*.txt")
        self.labels = []
        self.images = []
        self.val = val
        self.num_neg_val = 1800
        if not self.val:
            for label_file in label_files:
                with open(label_file, "r") as fp: annots = fp.read().splitlines()
                img_bboxes = []
                for annot in annots:
                    splits = annot.split(" ")
                    img_bboxes.append((max(0, int(splits[0])), max(0, int(splits[1])), min(int(splits[2]), 1250), min(int(splits[3]), 1250), splits[4]))

                self.labels.append(img_bboxes)
                self.images.append(label_file.replace(".txt", ".tif"))

            self.images.extend(glob.glob(f"{data_dir}/no/*.tif")[:self.num_neg_val])
        else:
            self.images.extend(glob.glob(f"{data_dir}/no/*.tif")[self.num_neg_val:])

            # Add unlabelled pool images for validation
            for pool_image in glob.glob(f"{data_dir}/yes/*.tif"):
                if pool_image not in self.images:
                    self.images.append(pool_image)
                    if len(self.images) >= 2*(2400-self.num_neg_val): break

        self.image_size = 417
        print(len(self.labels), len(self.images))

        self.flips = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
        ])
        self.rotation = v2.RandomRotation(degrees=[-180, 180])

        self.color_jitter = v2.ColorJitter(0.2, 0.2, 0.2, 0.05)

    def crop_pool_image(self, bbox, crop_size):
        if max(0, bbox[2]-crop_size) >= bbox[0]+1 or max(0, bbox[3]-crop_size) >= bbox[1]+1: crop_size = self.image_size
        lx = min(np.random.randint(max(0, bbox[2]-crop_size), bbox[0]+1), 1250-crop_size)
        ty = min(np.random.randint(max(0, bbox[3]-crop_size), bbox[1]+1), 1250-crop_size)

        return (lx, ty, lx+crop_size, ty+crop_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]))
        
        if not self.val:
            label = self.labels[index] if index < len(self.labels) else None
            if label == []:
                image = np.array(Image.open(self.images[np.random.randint(len(self.labels), len(self.images))]))
                label = None

            aug = True
            if label is not None:
                # Contains pool
                label = label[np.random.randint(len(label))]
                if aug:
                    if np.random.random() > 0.75:
                        crop_coords = self.crop_pool_image(label, self.image_size)
                    else:
                        crop_coords = self.crop_pool_image(label, int(self.image_size * (1 - np.random.random()/4)))
                else:
                    crop_coords = self.crop_pool_image(label, self.image_size)

                crop_image = image[crop_coords[0]:crop_coords[2], crop_coords[1]:crop_coords[3]]
                if aug:
                    crop_image = cv2.resize(crop_image, (self.image_size, self.image_size))

                pool = 1.
            else:
                # No pool
                if aug:
                    crop_size = self.image_size if np.random.random() > 0.75 else int(self.image_size * (1 - np.random.random()/4))
                    lx, ty = np.random.randint(0, 1250-crop_size), np.random.randint(0, 1250-crop_size)
                    crop_image = image[lx: lx+self.image_size, ty: ty+self.image_size].copy()
                    crop_image = cv2.resize(crop_image, (self.image_size, self.image_size))
                else:
                    lx, ty = np.random.randint(0, 1250-self.image_size), np.random.randint(0, 1250-self.image_size)
                    crop_image = image[lx: lx+self.image_size, ty: ty+self.image_size]
                
                pool = 0.

            img = torch.from_numpy(np.ascontiguousarray(crop_image)).float()
            img = img / 255
            img = img.movedim(2,0)

            img = self.flips(img)
            if np.random.random() < 0.75: img = self.rotation(img)
            img = self.color_jitter(img)

            return img, pool
        else:
            imgs = []
            for x in [0, 417, 833]:
                for y in [0, 417, 833]:
                    imgs.append(image[x:x+417, y:y+417])

            imgs = torch.from_numpy(np.ascontiguousarray(np.array(imgs))).float()
            imgs = imgs / 255
            imgs = imgs.movedim(3,1)

            pool = 1. if index > 2400-self.num_neg_val else 0.

            return imgs, pool
        


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, hex_keys, weekly_counts, target_fit_len=10):
        self.seq_len = 26
        self.hex_keys = hex_keys
        self.weekly_counts = weekly_counts # num_hexs, num_weeks

        self.add_neighbors = False
        self.target_fit_len = target_fit_len

    def __len__(self):
        return len(self.hex_keys)

    def __getitem__(self, index):
        seq_idx = np.random.randint(51 - (self.seq_len + 4))
        input_seq = []
        target_seq = self.weekly_counts[self.hex_keys[index]][seq_idx+self.seq_len+4 - self.target_fit_len: seq_idx+self.seq_len+4]
        slope, bias = np.polyfit(np.arange(self.target_fit_len), np.array(target_seq), 1)

        if self.add_neighbors:
            neighbor_seqs = []
            try:
                grid_disk = h3.grid_disk(self.hex_keys[index], 1)
            except:
                index = np.random.randint(len(self.hex_keys))
                target_seq = self.weekly_counts[self.hex_keys[index]][seq_idx+self.seq_len: seq_idx+self.seq_len+4]
                grid_disk = h3.grid_disk(self.hex_keys[index], 1)

            for hex_key in grid_disk:
                if hex_key in self.hex_keys:
                    n_idx = self.hex_keys.index(hex_key)
                    neighbor_seqs.append(self.weekly_counts[n_idx][seq_idx: seq_idx+self.seq_len])
                else:
                    neighbor_seqs.append([-1] * self.seq_len)

                input_seq.extend(neighbor_seqs[-1])
        else:
            input_seq = self.weekly_counts[self.hex_keys[index]][seq_idx: seq_idx+self.seq_len]
                    
        return torch.tensor(input_seq).float(), torch.tensor([slope, (self.target_fit_len - 4)*slope + bias]).float()