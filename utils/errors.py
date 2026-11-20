class PipelineError(Exception):
    """base class of the errors main.py reports without a traceback."""


class MissingInputError(PipelineError):
    """an input file of a stage is missing. the message names the command that writes it."""


class DataError(PipelineError):
    """input data exist but do not have the expected content."""


class ConfigError(PipelineError):
    """config/config.yaml is missing, unreadable or lacks a setting."""
