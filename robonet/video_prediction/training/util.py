import numpy as np
import matplotlib.pyplot as plt
import pdb

# def split_model_inference(real_images, params):
#     """
#     we use separate trajectories for the encoder than from the ones used for prediction training
#     :param inputs: dict with tensors in *time-major*
#     :param targets:dict with tensors in *time-major*
#     :return:
#     """
#     sbs = params['sub_batch_size']
#     bs = params['batch_size']
#     first_half = []
#     second_half = []
#     for i in range(bs // sbs):
#         first_half.append(real_images[sbs * i:sbs * i + sbs // 2])
#         second_half.append(real_images[sbs * i + sbs // 2:sbs * (i + 1)])
#     first_half = np.concatenate(first_half, 0)
#     second_half = np.concatenate(second_half, 0)
#     return first_half, second_half

def pad(real_frames, pad_amount):
    tensor = (real_frames * 255).astype(np.uint8)
    height_pad = np.zeros((tensor.shape[0], tensor.shape[1], pad_amount, tensor.shape[-2], tensor.shape[-1]), dtype=np.uint8)
    tensor = np.concatenate((height_pad, tensor, height_pad), axis=-3)
    width_pad = np.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], pad_amount, tensor.shape[-1]), dtype=np.uint8)
    tensor = np.concatenate((width_pad, tensor, width_pad), axis=-2)
    return tensor

def pad_and_concat(real_frames, pred_frames, pad_amount):
    real, pred = [(x * 255).astype(np.uint8) for x in (real_frames, pred_frames)]
    pred = np.concatenate([pred[:, 0][:, None] for _ in range(real.shape[1] - pred.shape[1])] + [pred], axis=1)
    image_summary_tensors = []
    for tensor in [real, pred]:
        height_pad = np.zeros((tensor.shape[0], tensor.shape[1], pad_amount, tensor.shape[-2], tensor.shape[-1]), dtype=np.uint8)
        tensor = np.concatenate((height_pad, tensor, height_pad), axis=-3)
        width_pad = np.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], pad_amount, tensor.shape[-1]), dtype=np.uint8)
        tensor = np.concatenate((width_pad, tensor, width_pad), axis=-2)
        image_summary_tensors.append(tensor)
    tensor = np.concatenate(image_summary_tensors, axis=2)
    return tensor


def render_dist(dist):
    rendered = np.zeros((dist.shape[0], dist.shape[1], dist.shape[2], dist.shape[3], 3), dtype=np.float32)
    for b in range(dist.shape[0]):
        for t in range(dist.shape[1]):
            rendered[b,t] = np.squeeze(plt.cm.viridis(dist[b][t])[:, :, :3])
    return rendered
