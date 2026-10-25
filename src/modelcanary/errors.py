"""Exception hierarchy. Every error raised on purpose by modelcanary derives from
:class:`ModelCanaryError` so callers can catch library failures in one place."""


class ModelCanaryError(Exception):
    """Base class for all modelcanary errors."""


class ConfigError(ModelCanaryError):
    """A policy, metric or scenario definition is invalid."""


class RolloutStateError(ModelCanaryError):
    """An operation is not allowed in the rollout's current state."""


class ModelNotFoundError(ModelCanaryError):
    """A model id is not present in the registry."""


class InferenceError(ModelCanaryError):
    """A model failed to produce a prediction."""


class AuditIntegrityError(ModelCanaryError):
    """The hash chain of an audit log does not verify."""
