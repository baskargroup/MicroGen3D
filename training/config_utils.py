def get_model_config(config, model_name):
    """
    Extract and validate configuration parameters for a given model.
    Raises errors if required parameters are missing or improperly formatted.
    """

    def get_int(key, default=0, allow_zero=True):
        value = model_cfg.get(key, default)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(f"'{model_name}.{key}' must be an integer, got: {value}")
        if not allow_zero and value <= 0:
            raise ValueError(f"'{model_name}.{key}' must be > 0, got: {value}")
        return value

    def get_float(key, default=0.0):
        value = model_cfg.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"'{model_name}.{key}' must be a float, got: {value}")

    def get_str_path(key):
        value = model_cfg.get(key, "")
        if value is None:
            return ""
        return str(value).strip()

    def get_dropout(key="dropout", default=0.1):
        value = model_cfg.get(key, default)
        try:
            value = float(value)
        except Exception:
            raise ValueError(f"'{model_name}.{key}' must be a float between 0 and 1, got: {value}")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"'{model_name}.{key}' must be in [0.0, 1.0], got: {value}")
        return value

    def get_bool(key, default=False):
        value = model_cfg.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ['true', 'yes', '1']:
                return True
            elif value.lower() in ['false', 'no', '0']:
                return False
        raise ValueError(f"'{model_name}.{key}' must be a boolean, got: {value}")

    model_cfg = config.get(model_name, {})

    return {
        'max_epochs': get_int('max_epochs', 0),
        'pretrained_path': get_str_path('pretrained_path'),
        'dropout': get_dropout(),
        'latent_dim_channels': get_int('latent_dim_channels', 1),
        'kld_loss_weight': get_float('kld_loss_weight', 1e-6),
        'n_feat': get_int('n_feat', 512, allow_zero=False),
        'timesteps': get_int('timesteps', 1000, allow_zero=False),
        'learning_rate': get_float('learning_rate', 1e-6),
        'stride1_first_layer': get_bool('stride1_first_layer', False),
        'max_channels': get_int('max_channels', 512, allow_zero=False),
    }


def get_context_indices(config):
    attributes = config.get("attributes", [])
    context_attributes = config.get("context_attributes") or attributes

    if not set(context_attributes).issubset(set(attributes)):
        raise ValueError(f"context_attributes must be a subset of attributes. Got: {context_attributes}")

    return [attributes.index(attr) for attr in context_attributes], context_attributes
