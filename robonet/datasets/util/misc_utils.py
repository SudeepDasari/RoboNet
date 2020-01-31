def override_dict(original, new):
    for k in new.keys():
        out = original.copy()
        out.update(new)
        return out
