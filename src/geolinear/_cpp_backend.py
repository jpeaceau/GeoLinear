"""
C++ backend loader for GeoLinear.

Attempts to import the compiled extension _geolinear_cpp and re-exports
GeoLinearConfig and CppGeoLinearRegressor.  If the extension is not built,
raises an ImportError with installation instructions.
"""

try:
    from geolinear._geolinear_cpp import (  # type: ignore[import]
        GeoLinearConfig,
        CppGeoLinearRegressor,
        CppGeoLinearClassifier,
        PartitionCoeffs,
    )
except ImportError as _e:
    raise ImportError(
        "GeoLinear C++ extension not found. Build with:\n"
        "    pip install .\n"
    ) from _e

__all__ = ["GeoLinearConfig", "CppGeoLinearRegressor", "CppGeoLinearClassifier", "PartitionCoeffs"]
