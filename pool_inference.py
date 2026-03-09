import numpy as np
import torch
import datetime
import glob
from PIL import Image
import cv2

from model import PoolClassifier

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('weights','','')
flags.DEFINE_string('image_file','','Set to image_file or directory')
flags.DEFINE_float('pool_thresh', 0.75, '')

def main(argv):
    classifier = PoolClassifier()
    if FLAGS.weights: classifier.load_state_dict(torch.load(FLAGS.weights, weights_only=True))
    #classifier.to("cuda")

    if "*" in FLAGS.image_file:
        image_files = glob.glob(FLAGS.image_file)
    else:
        image_files = [FLAGS.image_file]

    for image_file in image_files:
        image = np.array(Image.open(image_file))
        if image.shape[0] != 1250 or image.shape[1] != 1250:
            print("Images should be square and of size 1250x1250")
            image = cv2.resize(image, (1250, 1250))

        crops = []
        for x in [0, 417, 833]:
            for y in [0, 417, 833]:
                crops.append(image[x:x+417, y:y+417])

        img_batch = torch.from_numpy(np.ascontiguousarray(crops)).float()
        img_batch = img_batch / 255
        img_batch = img_batch.movedim(3, 1)

        result = classifier(img_batch).sigmoid() # 9
        pool_conf, pool_region = result.max(dim=0)

        image = image.astype(np.uint8)

        if pool_conf > FLAGS.pool_thresh:
            region_x, region_y = int(pool_region % 3), int(pool_region // 3)
            cv2.rectangle(image, (region_x*417, region_y*417), ((region_x+1)*417, (region_y+1)*417), (255, 0, 0), 2)
            cv2.putText(image, f"{pool_conf}", (region_x*417, region_y*417), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Detected pool on {image_file} with confidence {pool_conf}")
        else:
            print(f"No pool detected on {image_file}, Max detection confidence: {pool_conf}")

        cv2.imwrite(f"{image_file[:-4]}_result.png", image[:,:,::-1])



if __name__ == '__main__':
    app.run(main)



