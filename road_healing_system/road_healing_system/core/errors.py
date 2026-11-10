"""Exception types with user-readable messages."""


class RoadHealingError(Exception):
    """Base class for every error raised deliberately by the pipeline."""


class ImageLoadError(RoadHealingError):
    """The input image is missing, unsupported, empty or cannot be decoded."""


class MaskError(RoadHealingError):
    """A supplied road or reference mask is unusable."""


class ConfigError(RoadHealingError, ValueError):
    """A configuration value is out of range or malformed."""


class StageOrderError(RoadHealingError):
    """A stage was run before the stages it depends on."""


class ProcessingError(RoadHealingError):
    """A processing stage failed in a way the user should be told about."""
