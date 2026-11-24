from app.core.exceptions import ApplicationError


class InvalidPricingParamsError(ApplicationError):
    status_code = 400
    code = "INVALID_PRICING_PARAMS"
    message = "product_id is required"


class MissingPricingParamsError(ApplicationError):
    status_code = 400
    code = "MISSING_PRICING_PARAMS"
    message = "Missing required pricing parameters"


class PricingBindingNotFoundError(ApplicationError):
    status_code = 404
    code = "PRICING_BINDING_NOT_FOUND"
    message = "Pricing binding not found"
