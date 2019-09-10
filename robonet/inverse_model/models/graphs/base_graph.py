from robonet.video_prediction.models.graphs.base_graph import BaseGraph as BaseVpredGraph
import tensorflow as tf


class BaseGraph(BaseVpredGraph):
    @staticmethod
    def default_hparams():
        return {
        }
