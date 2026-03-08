import numpy as np
import cv2
import glob
import os
#from win32gui import GetWindowText, GetForegroundWindow

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_bool("bbox_persist", False, "")
flags.DEFINE_integer("grid", 35, "")

mouse_down = (0,0)
mouse_up = (0,0)
busy_drawing = False
image = None
clone = []
bounding_boxes = []
digits = [str(int("0")+d) for d in range(10)]
with open("categories.txt", "r") as fp: cats = fp.read().splitlines()
np.random.seed(10)
colors = [(255,0,0), (0,255,0), (0,0,255), (255,0,255), (122, 122, 255), (122, 255, 122)]
for cat in range(len(cats)-len(colors)):
    colors.append((np.random.randint(255), np.random.randint(255), np.random.randint(255)))

cat_ix = 0

def onClick(event, x, y, cv_flags, param):
    global mouse_down
    global mouse_up
    global busy_drawing
    global image
    global clone
    global bounding_boxes
    global cats
    global cat_ix
    global digits

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = (x, y)
        busy_drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_up = (x,y)
        #if GetWindowText(GetForegroundWindow()) in cats:
        image = clone[-1].copy()
        busy_drawing = False
        if FLAGS.grid > 1 and cats[cat_ix] in ["gridcell", "rowcell"]:
            curr_y = mouse_down[1]
            cell_height = (mouse_up[1] - mouse_down[1]) / FLAGS.grid
            for gr_ix in range(FLAGS.grid):
                bounding_boxes.append((mouse_down[0], round(curr_y), mouse_up[0], round(curr_y + cell_height), cats[cat_ix]))
                cv2.rectangle(image, (mouse_down[0], round(curr_y)), (mouse_up[0], round(curr_y + cell_height)), colors[cat_ix], 1)
                curr_y += cell_height

            if cats[cat_ix] == "rowcell":
                start_row = input("Start row: ")
                for b_ix in range(len(bounding_boxes)):
                    bbox = bounding_boxes[b_ix]
                    if bbox[4] == "rowcell":
                        digit_width = (bbox[2] - bbox[0]) / len(start_row)
                        curr_x = bbox[0]
                        for c in start_row:
                            bounding_boxes.append((int(curr_x), bbox[1], round(curr_x + digit_width), bbox[3], c))
                            cv2.rectangle(image, (int(curr_x), bbox[1]), (round(curr_x + digit_width), bbox[3]), colors[cat_ix], 1)
                            cv2.putText(image, c, (int(curr_x)+3, bbox[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
                            curr_x += digit_width

                        start_row = str(int(start_row) + 1)

            clone.append(image.copy())
        else:
            cv2.rectangle(image, mouse_down, mouse_up, colors[cat_ix], 1)
            cv2.putText(image, cats[cat_ix], (mouse_down[0]+3, mouse_down[1]+1*cat_ix), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA)
            clone.append(image.copy())
            bounding_boxes.append((mouse_down[0], mouse_down[1], mouse_up[0], mouse_up[1], cats[cat_ix]))

    elif event == cv2.EVENT_MOUSEMOVE:
        if busy_drawing:
            image = clone[-1].copy()
            cv2.rectangle(image, mouse_down, (x,y), colors[cat_ix], 1)
        else:
            image = clone[-1].copy()
            cv2.line(image, (x, 0), (x, image.shape[0]), (122,122,122), 1)
            cv2.line(image, (0, y), (image.shape[1], y), (122,122,122), 1)

def main(argv):
    global mouse_down
    global mouse_up
    global busy_drawing
    global image
    global clone
    global bounding_boxes
    global cats
    global cat_ix
    global digits

    images = glob.glob(f"C:\\Users\\david\\ds_code_challenge\\swimming_pools\\yes\\*.TIF")
    print(len(images))
    count = 0
    resize_ratio = 2

    while count < len(images):
        img_name = images[count]
        print(count)
        count += 1
        image = cv2.imread(img_name)
        if image is None: continue
        image = cv2.resize(image, (int(image.shape[1]//resize_ratio), int(image.shape[0]//resize_ratio)))
        if os.path.exists(f"{img_name[:-4]}.txt"):
            with open(f"{img_name[:-4]}.txt", "r") as fp: annots = fp.read().splitlines()
            bounding_boxes = []
            for annot in annots:
                x1, y1, x2, y2, cat = annot.split(" ")
                bounding_boxes.append((round(int(x1) / resize_ratio), round(int(y1) / resize_ratio), round(int(x2) / resize_ratio), round(int(y2) / resize_ratio), cat))

        clone = []
        clone.append(image.copy())
        missing_digits = False
        if FLAGS.bbox_persist or os.path.exists(f"{img_name[:-4]}.txt"):
            print(bounding_boxes)
            if not os.path.exists(f"{img_name[:-4]}.txt"):
                for bb_ix in range(len(bounding_boxes)-1, -1, -1):
                    if bounding_boxes[bb_ix][4] in digits: del bounding_boxes[bb_ix]

            grid_count = None
            drawn_digits = False
            for bbox in bounding_boxes:
                color = (255, 0, 255) if bbox[4] in digits else colors[cats.index(bbox[4])]
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
                if bbox[4] in ["gridcell", "rowcell"] and FLAGS.grid > 1:
                    if bbox[4] == "rowcell" and not os.path.exists(f"{img_name[:-4]}.txt"): missing_digits = True
                    if grid_count is None: grid_count = 1
                    elif grid_count < FLAGS.grid-1: grid_count += 1
                    else:
                        grid_count = None
                        if bbox[4] in ["gridcell"]: clone.append(image.copy())
                elif bbox[4] in digits:
                    cv2.putText(image, bbox[4], (bbox[0]+3, bbox[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
                    drawn_digits = True
                else:
                    cv2.putText(image, bbox[4], (bbox[0]+3, bbox[1]+1*cats.index(bbox[4])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA)
                    clone.append(image.copy())

            if drawn_digits: clone.append(image.copy())
        else:
            bounding_boxes = []

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", onClick)

        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", image)
            cv2.setWindowTitle("image", cats[cat_ix])
            key = cv2.waitKey(1) & 0xFF

            if missing_digits:
                start_row = input("Start row: ")
                for b_ix in range(len(bounding_boxes)):
                    bbox = bounding_boxes[b_ix]
                    if bbox[4] == "rowcell":
                        digit_width = (bbox[2] - bbox[0]) / len(start_row)
                        curr_x = bbox[0]
                        for c in start_row:
                            bounding_boxes.append((int(curr_x), bbox[1], round(curr_x + digit_width), bbox[3], c))
                            cv2.rectangle(image, (int(curr_x), bbox[1]), (round(curr_x + digit_width), bbox[3]), (255, 0, 255), 1)
                            cv2.putText(image, c, (int(curr_x)+3, bbox[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
                            curr_x += digit_width

                        start_row = str(int(start_row) + 1)

                clone.append(image.copy())
                missing_digits = False

            if key == ord("n"):
                with open(f"{img_name[:-4]}.txt","w") as fp:
                    for bbox in bounding_boxes:
                        if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4: continue
                        fp.write('{} {} {} {} {}\n'.format(round(bbox[0]*resize_ratio), round(bbox[1]*resize_ratio), round(bbox[2]*resize_ratio), round(bbox[3]*resize_ratio), bbox[4]))
                break
            elif key == ord("d"):
                if len(bounding_boxes) > 0:
                    while bounding_boxes[-1][4] in digits: del bounding_boxes[-1]

                    if bounding_boxes[-1][4] in ["gridcell", "rowcell"] and FLAGS.grid > 1:
                        del clone[-1]
                        for gr in range(FLAGS.grid):
                            del bounding_boxes[-1]
                    else:
                        del clone[-1]
                        del bounding_boxes[-1]
            elif key == ord("c"):
                cat_ix += 1
                if cat_ix == len(cats): cat_ix = 0
            elif key == ord("b"):
                if os.path.exists(f"{images[count-1][:-4]}.txt"): os.remove(f"{images[count-1][:-4]}.txt")
                count -= 2
                break
            elif key == ord("q"):
                break

        if key == ord("q"): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(main)