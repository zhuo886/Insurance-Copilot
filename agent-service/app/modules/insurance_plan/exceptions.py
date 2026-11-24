from app.core.exceptions import ApplicationError


class InsurancePlanNotFoundError(ApplicationError):
    status_code = 404
    code = "INSURANCE_PLAN_NOT_FOUND"
    message = "保险配置方案不存在或无权访问"


class InsurancePlanItemNotFoundError(ApplicationError):
    status_code = 404
    code = "PLAN_ITEM_NOT_FOUND"
    message = "保险方案产品项不存在"
