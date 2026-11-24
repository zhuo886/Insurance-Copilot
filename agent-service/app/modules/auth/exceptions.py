from app.core.exceptions import ApplicationError


class InvalidRegisterRequestError(ApplicationError):
    status_code = 400
    code = "INVALID_REGISTER_REQUEST"
    message = "用户名为空、邮箱格式错误或密码少于 6 位"


class UserExistsError(ApplicationError):
    status_code = 409
    code = "USER_EXISTS"
    message = "用户名或邮箱已存在"


class InvalidCredentialsError(ApplicationError):
    status_code = 400
    code = "INVALID_CREDENTIALS"
    message = "用户名/邮箱或密码错误"


class InvalidRefreshTokenError(ApplicationError):
    status_code = 401
    code = "INVALID_REFRESH_TOKEN"
    message = "Refresh token is invalid or expired"


class UserNotFoundError(ApplicationError):
    status_code = 404
    code = "USER_NOT_FOUND"
    message = "User not found"


class InactiveUserError(ApplicationError):
    status_code = 400
    code = "INACTIVE_USER"
    message = "用户已被停用"
