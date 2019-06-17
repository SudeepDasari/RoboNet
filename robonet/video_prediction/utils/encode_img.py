import imageio
import io
import cv2
import numpy as np


def construct_image_tile(tensor):
    assert len(tensor.shape) == 4 or len(tensor.shape) == 5, "assumes (B, H, W, C) or (B, T, H, W, C) tensor"
    return np.concatenate([im for im in tensor], axis=-2)


def encode_images(tensor):
    if len(tensor.shape) == 3:
        return cv2.imencode('.jpg', tensor[:, :, ::-1])[1]
    elif len(tensor.shape) == 4:
        buffer = io.BytesIO()
        writer = imageio.get_writer(buffer, format='gif')
        [writer.append_data(im) for im in tensor]
        writer.close()
        buffer.seek(0)
        return buffer.read()
    raise NotImplementedError
