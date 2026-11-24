from app.core.exceptions import ApplicationError


class ProductNotFoundError(ApplicationError):
    status_code = 404
    code = "PRODUCT_NOT_FOUND"
    message = "Product not found"


class PricingSchemaNotFoundError(ApplicationError):
    status_code = 404
    code = "PRICING_SCHEMA_NOT_FOUND"
    message = "Pricing schema not found"
