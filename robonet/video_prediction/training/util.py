import numpy as np
import matplotlib.pyplot as plt


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
    return np.concatenate(image_summary_tensors, axis=2)


def expected_pixel_distance(real_dist, pred_dist):
    import pdb; pdb.set_trace()
    return 0


def render_dist(dist):
    rendered = np.zeros((dist.shape[0], dist.shape[1], dist.shape[2], dist.shape[3], 3), dtype=np.uint8)
    for b in range(dist.shape[0]):
        for t in range(dist.shape[1]):
            dist_t = dist[b][t] / (np.amax(dist_p[b][t]) + 1e-6)
            rendered[b,t] = (np.squeeze(plt.cm.viridis(dist_t)[:, :, :3]) * 255).astype(np.uint8)
    return rendered
