"""
This Module provides functions to apply data augmentation 
"""

from torchvision import transforms

AUGMENTATIONS = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "resize_scale": (0.7, 1.0),
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.2,
    "color_jitter_prob": 0.8,
    "gray_scale_prob": 0.2,
}

# Data loader
def data_augmentation(img, augment: dict = None):

    """
    Data augmentation such as vertical and horizontal flip,
    random rotation and random sized crop.

    Arguments:
    ----------
        img: 3D numpy array
            input image with shape (C,H,W)
        transforms: dictionary
            kwargs to transform images
    """

    augment = augment or AUGMENTATIONS

    # Defines Color Jitter
    color_jitter = transforms.ColorJitter(
        brightness=augment["brightness"],
        contrast=augment["contrast"],
        saturation=augment["saturation"],
        hue=augment["hue"],
    )

    # Defines data augmentation
    augmentation = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=augment["vertical_flip_prob"]),
            transforms.RandomVerticalFlip(p=augment["horizontal_flip_prob"]),
            transforms.RandomResizedCrop(size=img.shape[1], scale=augment["resize_scale"]),
            transforms.RandomApply([color_jitter], p=augment["color_jitter_prob"]),
            transforms.RandomGrayscale(p=augment["gray_scale_prob"]),
        ]
    )

    img = augmentation(img)

    return img
