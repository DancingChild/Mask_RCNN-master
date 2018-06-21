import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448
    IMAGE_SHAPE = [448, 448, 3]

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = ShapesConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


from PIL import Image


class SpaceNetDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def load_shapes(self, img_list, img_floder, mask_floder):
        """Generate the requested number of synthetic images.
        #count: number of images to generate.
        image_id_list : list of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "shapes")
        # self.add_class("shapes", 2, "others")
        # self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        # imglist = os.listdir(img_floder)
        for i in imglist:
            # 获取图片宽和高
            # filestr = imglist[i]
            image_id = i.split("_")[-1][3:-4]
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + "mask_" + i
            img_path = img_floder + "/" + i
            # yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            # print(i)
            cv_img = cv2.imread(img_path)
            self.add_image("shapes", image_id=image_id, path=img_path,
                           width=cv_img.shape[1], height=cv_img.shape[0],
                           mask_path=mask_path)

    def draw_mask(self, num_obj, mask, image, image_id):
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    #     def load_image(self, image_id):
    #         """Generate an image from the specs of the given image ID.
    #         Typically this function loads the image from a file, but
    #         in this case it generates the image on the fly from the
    #         specs in image_info.
    #         """
    # #         info = self.image_info[image_id]
    # #         bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
    # #         image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
    # #         image = image * bg_color.astype(np.uint8)
    # #         for shape, color, dims in info['shapes']:
    # #             image = self.draw_shape(image, shape, dims, color)
    #         img_dir = "/home/ly/data/dl_data/spacenet/AOI_5_Khartoum_Train/RGB-PanSharpen"
    #         img_name = "RGB-PanSharpen_AOI_5_Khartoum_img{}.tif".format(image_id)
    #         img_path = os.path.join(img_dir, img_name)
    #         if not os.path.isfile(img_path):
    #             pass
    #         else:
    #             image = cv2.imread(img_path)
    #             return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        # print("image_id",image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        # print("num_obj:",num_obj)
        mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        # print(mask.shape)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 1, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # labels = []
        #         labels = self.from_yaml_get_class(image_id)
        #         labels_form = []
        #         for i in range(len(labels)):
        #             if labels[i].find("tongue") != -1:
        #                 # print "box"
        #                 labels_form.append("tongue")
        #         class_ids = np.array([self.class_names.index(s) for s in labels_form])
        class_ids = np.array([1])

        return mask, class_ids.astype(np.int32)

img_dir = "/home/ly/data/dl_data/spacenet/AOI_5_Khartoum_Train/RGB-PanSharpen"
mask_dir = "/home/ly/data/dl_data/spacenet/AOI_5_Khartoum_Train/mask"
img_id_list = [ ]
imglist = os.listdir(img_dir)
for i in imglist:
    img_id = i.split('_')[-1][3:-4]
    img_id_list.append(img_id)


dataset_train = SpaceNetDataset()
dataset_train.load_shapes(imglist[:-50],img_dir, mask_dir)
dataset_train.prepare()

#Validation dataset
dataset_val = SpaceNetDataset()
dataset_val.load_shapes(imglist[-50:], img_dir, mask_dir)
dataset_val.prepare()

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=0.01,
            epochs=1,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=0.01 / 10,
            epochs=5,
            layers="all")
