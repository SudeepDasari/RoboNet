def get_models(class_name):
    if class_name == 'VAEInverse':
        from .vae_inverse_model import VAEInverse
        return VAEInverse
