from robonet.video_prediction.models.base_model import BaseModel
from robonet.inverse_model.models.graphs import get_graph_class


class BaseInverseModel(BaseModel):
    def _get_graph(self, graph_type):
        return get_graph_class(graph_type)
