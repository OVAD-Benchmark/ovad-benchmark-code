from .attribute_classifier import AttributeClassifier


def build_attribute_predictor(cfg, input_shape):
    """
    Build a attribute head.
    """
    return AttributeClassifier(cfg, input_shape)
