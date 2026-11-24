from app.core.exceptions import ApplicationError


class ClaimNotFoundError(ApplicationError):
    status_code = 404
    code = "CLAIM_NOT_FOUND"
    message = "Claim not found"


class ApplicationFormNotFoundError(ApplicationError):
    status_code = 404
    code = "APPLICATION_FORM_NOT_FOUND"
    message = "Application form not found"
