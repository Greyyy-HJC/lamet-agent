"""Custom exception types used across the lamet-agent package."""


class LametAgentError(Exception):
    """Base class for package-specific errors."""


class ManifestValidationError(LametAgentError):
    """Raised when a manifest is missing required data or contains invalid values."""


class KernelLoadError(LametAgentError):
    """Raised when an inline kernel cannot be compiled or validated."""


class WorkflowResolutionError(LametAgentError):
    """Raised when the planner cannot resolve a valid workflow."""


class StageExecutionError(LametAgentError):
    """Raised when a stage fails to complete its work."""


class OptionalDependencyError(LametAgentError):
    """Raised when an optional runtime dependency is required but unavailable."""


class OAuthConfigurationError(LametAgentError):
    """Raised when OAuth provider configuration is incomplete or invalid."""


class OAuthFlowError(LametAgentError):
    """Raised when an OAuth authorization or token exchange flow fails."""


class TokenStoreError(LametAgentError):
    """Raised when persisted OAuth tokens cannot be loaded or saved."""
