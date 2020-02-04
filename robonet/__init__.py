try:
    from robonet.video_prediction.training import GIFLogger
    from robonet.video_prediction.training import get_trainable as vpred_trainable
    from robonet.inverse_model.training import get_trainable as inverse_trainable
    from robonet.awr.training import get_trainable as awr_trainable
except:
    print('could not import trainables!')


def get_trainable(class_name):
    available_trainables = [vpred_trainable, inverse_trainable, awr_trainable]
    for a in available_trainables:
        try:
            return a(class_name)
        except NotImplementedError:
            pass
    raise NotImplementedError
