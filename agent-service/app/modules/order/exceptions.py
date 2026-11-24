from app.core.exceptions import ApplicationError


class OrderNotFoundError(ApplicationError):
    status_code = 404
    code = "ORDER_NOT_FOUND"
    message = "Order not found"


class QuoteNotFoundError(ApplicationError):
    status_code = 404
    code = "QUOTE_NOT_FOUND"
    message = "Quote not found"
