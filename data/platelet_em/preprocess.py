from PIL import Image
import numpy as np
from tqdm import tqdm
import os


def class_to_rgb(class_array):
    """
    transforms a one-hot aray into an RGB segmentation image
    Parameters:
        class_array: np.array of size H x W
    Returs:
        PIL image of size H x W
    """
    # create rgb image
    rgb_img = np.zeros((*class_array.shape[:2], 3), dtype=np.uint8)
    class_colors = [(0, 0, 0), (0, 40, 255), (0, 212, 255)]

    # run over image
    for h in range(class_array.shape[0]):
        for w in range(class_array.shape[1]):
            c = class_array[h, w]
            color = class_colors[c]
            rgb_img[h, w, :] = color
    return Image.fromarray(rgb_img)


def preprocess_tiff_stack(
    intensity_file,
    class_file,
    from_slice,
    to_slice,
    crop_bbox,
    intensity_output_file,
    class_output_file,
    semantic_output_file,
    topology_appear_file=None,
    topology_disappear_file=None,
    topology_appear_output_file=None,
    topology_disappear_output_file=None,
    topology_combined_output_file=None,
):
    """
    Preprocesses the platelet-em dataset
    combines organelle classes
    crops a section

    Args:
        intensity_file ([type]): [description]
        class_file ([type]): [description]
        from_slice ([type]): [description]
        to_slice ([type]): [description]
        crop_bbox ([type]): [description]
        intensity_output_file ([type]): [description]
        class_output_file ([type]): [description]
        semantic_output_file ([type]): [description]
    """
    # load data and initialize output lists
    intensity_image_stack = Image.open(intensity_file)
    class_image_stack = Image.open(class_file)
    intensity_image_stack_output = []
    class_image_stack_output = []
    semantic_image_stack_output = []

    has_topology_annotation = topology_appear_file is not None
    if has_topology_annotation:
        topology_appear_image_stack = Image.open(topology_appear_file)
        topology_disappear_image_stack = Image.open(topology_disappear_file)
        topology_appear_output = []
        topology_disappear_output = []
        topology_combined_output = []

    # cut into slices
    for i in range(from_slice, to_slice):
        # load slice
        intensity_image_stack.seek(i)
        intensity_image = np.array(intensity_image_stack)
        class_image_stack.seek(i)
        class_image = np.array(class_image_stack)
        if has_topology_annotation:
            topology_appear_image_stack.seek(i)
            topology_appear_image = np.array(topology_appear_image_stack)
            topology_disappear_image_stack.seek(i)
            topology_disappear_image = np.array(topology_disappear_image_stack)
            if i == 0:
                topology_combined_image = np.zeros_like(
                    topology_disappear_image)
            else:
                topology_appear_image_stack.seek(i-1)
                topology_combined_image = np.clip(
                    np.array(topology_appear_image_stack) + topology_appear_image, 0, 1)

        # crop bbox
        intensity_image = intensity_image[crop_bbox[0]
            :crop_bbox[2], crop_bbox[1]: crop_bbox[3]]
        class_image = class_image[crop_bbox[0]
            :crop_bbox[2], crop_bbox[1]: crop_bbox[3]]
        if has_topology_annotation:
            topology_appear_image = topology_appear_image[crop_bbox[0]
                :crop_bbox[2], crop_bbox[1]: crop_bbox[3]]
            topology_disappear_image = topology_disappear_image[crop_bbox[0]
                :crop_bbox[2], crop_bbox[1]: crop_bbox[3]]
            topology_combined_image = topology_combined_image[crop_bbox[0]
                :crop_bbox[2], crop_bbox[1]: crop_bbox[3]]

        # map classes
        # reduce classes. Map all labels >1 to 2
        class_image[class_image > 1] = 2

        # build output
        intensity_image_stack_output.append(Image.fromarray(intensity_image))
        class_image_stack_output.append(Image.fromarray(class_image))
        semantic_image_stack_output.append(
            class_to_rgb(class_image))  # re-generate rgb images
        if has_topology_annotation:
            topology_appear_output.append(
                Image.fromarray(topology_appear_image))
            topology_disappear_output.append(
                Image.fromarray(topology_disappear_image))
            topology_combined_output.append(
                Image.fromarray(topology_combined_image))

    # save result
    intensity_image_stack_output[0].save(
        intensity_output_file, save_all=True, append_images=intensity_image_stack_output[1:]
    )
    class_image_stack_output[0].save(
        class_output_file, save_all=True, append_images=class_image_stack_output[1:]
    )
    semantic_image_stack_output[0].save(
        semantic_output_file, save_all=True, append_images=semantic_image_stack_output[1:]
    )
    if has_topology_annotation:
        topology_appear_output[0].save(
            topology_appear_output_file, save_all=True, append_images=topology_appear_output[1:]
        )
        topology_disappear_output[0].save(
            topology_disappear_output_file, save_all=True, append_images=topology_disappear_output[1:]
        )
        topology_combined_output[0].save(
            topology_combined_output_file, save_all=True, append_images=topology_combined_output[1:]
        )


if __name__ == "__main__":
    os.makedirs("./data/platelet_em/train/image/", exist_ok=True)
    os.makedirs("./data/platelet_em/train/label/", exist_ok=True)
    os.makedirs("./data/platelet_em/train/semantic/", exist_ok=True)
    os.makedirs("./data/platelet_em/train/topology/", exist_ok=True)
    os.makedirs("./data/platelet_em/val/image/", exist_ok=True)
    os.makedirs("./data/platelet_em/val/label/", exist_ok=True)
    os.makedirs("./data/platelet_em/val/semantic/", exist_ok=True)
    os.makedirs("./data/platelet_em/val/topology_appear/", exist_ok=True)
    os.makedirs("./data/platelet_em/val/topology_disappear/", exist_ok=True)
    os.makedirs("./data/platelet_em/val/topology_combined/", exist_ok=True)
    os.makedirs("./data/platelet_em/test/image/", exist_ok=True)
    os.makedirs("./data/platelet_em/test/label/", exist_ok=True)
    os.makedirs("./data/platelet_em/test/semantic/", exist_ok=True)
    os.makedirs("./data/platelet_em/test/topology_appear/", exist_ok=True)
    os.makedirs("./data/platelet_em/test/topology_disappear/", exist_ok=True)
    os.makedirs("./data/platelet_em/test/topology_combined/", exist_ok=True)
    bboxes = [(16, 16, 272, 272),  # bounding boxes to split 800x800 image into 9x 256x256 patches
              (16, 272, 272, 528),
              (16, 528, 272, 784),
              (272, 16, 528, 272),
              (272, 272, 528, 528),
              (272, 528, 528, 784),
              (528, 16, 784, 272),
              (528, 272, 784, 528),
              (528, 528, 784, 784), ]

    # preprocess training data
    for p in tqdm(range(9), desc="extracting training images"):
        preprocess_tiff_stack(
            "./data/platelet_em/raw/images/50-images.tif",
            "./data/platelet_em/raw/labels-class/50-class.tif",
            0,
            50,
            bboxes[p],
            f"./data/platelet_em/train/image/{p}.tif",
            f"./data/platelet_em/train/label/{p}.tif",
            f"./data/platelet_em/train/semantic/{p}.tif",
        )
    # preprocess validaion data
    for p in tqdm(range(4), desc="extracting validation images"):
        preprocess_tiff_stack(
            "./data/platelet_em/raw/images/24-images.tif",
            "./data/platelet_em/raw/labels-class/24-class.tif",
            0,
            24,
            bboxes[p],
            f"./data/platelet_em/val/image/{p}.tif",
            f"./data/platelet_em/val/label/{p}.tif",
            f"./data/platelet_em/val/semantic/{p}.tif",
            "./data/platelet_em/raw/labels-topology/topology_change_appear.tiff",
            "./data/platelet_em/raw/labels-topology/topology_change_disappear.tiff",
            f"./data/platelet_em/val/topology_appear/{p}.tif",
            f"./data/platelet_em/val/topology_disappear/{p}.tif",
            f"./data/platelet_em/val/topology_combined/{p}.tif",
        )
    # preprocess test data
    for p in tqdm(range(4, 9), desc="extracting test images"):
        preprocess_tiff_stack(
            "./data/platelet_em/raw/images/24-images.tif",
            "./data/platelet_em/raw/labels-class/24-class.tif",
            0,
            24,
            bboxes[p],
            f"./data/platelet_em/test/image/{p}.tif",
            f"./data/platelet_em/test/label/{p}.tif",
            f"./data/platelet_em/test/semantic/{p}.tif",
            "./data/platelet_em/raw/labels-topology/topology_change_appear.tiff",
            "./data/platelet_em/raw/labels-topology/topology_change_disappear.tiff",
            f"./data/platelet_em/test/topology_appear/{p}.tif",
            f"./data/platelet_em/test/topology_disappear/{p}.tif",
            f"./data/platelet_em/test/topology_combined/{p}.tif",
        )
