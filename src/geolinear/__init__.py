"""
GeoLinear — boosted linear models on HVRT cooperative geometry.

Core classes
------------
GeoLinear           Boosted partition-local Ridge regressor.
GeoLinearClassifier Boosted logistic classifier.
TSQTransformer      Sklearn-compatible T, S, Q feature augmentation transformer.

Utilities
---------
augment_TSQ         Append cooperative statistics T, S, Q to any feature matrix.
"""

from geolinear.regressor import GeoLinear, GeoLinearClassifier
from geolinear.augment   import augment_TSQ, TSQTransformer

__version__ = "0.3.0"

__all__ = [
    "GeoLinear",
    "GeoLinearClassifier",
    "augment_TSQ",
    "TSQTransformer",
]
