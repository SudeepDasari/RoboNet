"""
Simple script that shows the video-predictor API in action
"""


from robonet.video_prediction.testing.model_evaluation_interface import VPredEvaluation
import numpy as np

test_hparams = {}
test_hparams['designated_pixel_count'] = 1                  # number of selected pixels
test_hparams['run_batch_size'] = 200                        # number of predictions run through model concurrently
N_ACTIONS = 300                                             # total actions to predict: can be different from run_batch_size!

# feed in restore path and test specific hyperparams
model = VPredEvaluation('~/Downloads/franka_sanity/sanity_check_model/checkpoint_170000', test_hparams)
model.restore()

# context tensors needed for prediction
context_tensors = {}
context_tensors['context_actions'] = np.zeros((model.n_context - 1, model.adim))
context_tensors['context_states'] = np.zeros((model.n_context, model.sdim))                              # not needed for all models
height, width = model.img_size
context_tensors['context_frames'] = np.zeros((model.n_context, model.n_cam, height, width, 3))           # inputs should be RGB float \in [0, 1]
context_tensors['context_pixel_distributions'] = np.zeros((model.n_context, model.n_cam, height,         # spatial disributions (sum across image should be 1)
                                                            width, test_hparams['designated_pixel_count']))
context_tensors['context_pixel_distributions'][:, :, 24, 32, :] = 1.0

# actions for frames to be predicted
action_tensors = {}
action_tensors['actions'] = np.zeros((N_ACTIONS, model.horizon, model.adim))

results = model(context_tensors, action_tensors)
predicted_frames = results['predicted_frames']                        # RGB images, shape (N_ACTIONS, HORIZON, N_CAMS, 48, 64, 3)
predicted_distributions = results['predicted_pixel_distributions']    # pixel distributions, shape (N_ACTIONS, HORIZON, N_CAMS, 48, 64, designated_pixel_count)
print('predicted_frames has shape', predicted_frames.shape)
