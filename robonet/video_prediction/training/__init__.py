from .ray_util.gif_logger import GIFLogger


def get_trainable(class_name):
    if class_name == 'VPredTrainable':
        from .trainable_interface import VPredTrainable
        return VPredTrainable
    if class_name == 'BalancedCamFilter':
        from .data_filter import BalancedCamFilter
        return BalancedCamFilter
    if class_name == 'RobotSetFilter':
        from .data_filter import RobotSetFilter
        return RobotSetFilter
    raise NotImplementedError
