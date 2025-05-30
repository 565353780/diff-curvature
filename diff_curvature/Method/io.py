import numpy as np
from PIL import Image


def save_image(img, path):
    img = img.cpu().numpy()
    img = img * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
