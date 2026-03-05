"""Domain exceptions for Lya."""


class DomainException(Exception):
    """Base exception for domain errors."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or "DOMAIN_ERROR"


class ValidationError(DomainException):
    """Raised when domain validation fails."""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field


class BusinessRuleError(DomainException):
    """Raised when a business rule is violated."""

    def __init__(self, message: str, rule: str | None = None) -> None:
        super().__init__(message, "BUSINESS_RULE_ERROR")
        self.rule = rule


class EntityNotFoundError(DomainException):
    """Raised when an entity is not found."""

    def __init__(self, entity_type: str, entity_id: str) -> None:
        message = f"{entity_type} with id '{entity_id}' not found"
        super().__init__(message, "ENTITY_NOT_FOUND")
        self.entity_type = entity_type
        self.entity_id = entity_id


class StateTransitionError(DomainException):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, entity: str, from_state: str, to_state: str) -> None:
        message = f"Cannot transition {entity} from {from_state} to {to_state}"
        super().__init__(message, "INVALID_STATE_TRANSITION")
        self.from_state = from_state
        self.to_state = to_state


class AuthorizationError(DomainException):
    """Raised when an operation is not authorized."""

    def __init__(self, message: str = "Operation not authorized") -> None:
        super().__init__(message, "AUTHORIZATION_ERROR")
