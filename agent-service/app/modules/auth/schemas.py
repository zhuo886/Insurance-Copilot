from pydantic import BaseModel, ConfigDict


class RegisterRequest(BaseModel):
    username: str = ""
    email: str = ""
    password: str = ""


class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    display_name: str | None = None


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    user: UserResponse


class LogoutResponse(BaseModel):
    ok: bool = True
