"""用户身份解析：从 Java 后端签发的 JWT 中解析 openid / userId。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import jwt

# 课程项目的 JWT 密钥较短（与 Java 配置一致），屏蔽 PyJWT 的密钥长度告警
warnings.filterwarnings("ignore", message="The HMAC key is")

from .config import USER_JWT_SECRET


@dataclass
class AuthUser:
    user_id: Optional[str]      # 优先 openid；没有 openid 时用 user_<userId>
    openid: Optional[str]
    raw_user_id: Optional[str]
    valid: bool


def _extract_token(token: Optional[str], authorization: Optional[str]) -> Optional[str]:
    if token and token.strip():
        return token.strip()
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None


def resolve_user(token: Optional[str] = None, authorization: Optional[str] = None) -> AuthUser:
    """解析 JWT；无效或缺失时返回匿名用户，不抛异常。"""
    raw_token = _extract_token(token, authorization)
    if not raw_token:
        return AuthUser(user_id=None, openid=None, raw_user_id=None, valid=False)

    try:
        payload = jwt.decode(
            raw_token,
            USER_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except jwt.ExpiredSignatureError:
        print("⚠️ 用户 token 已过期")
        return AuthUser(user_id=None, openid=None, raw_user_id=None, valid=False)
    except Exception as e:
        print(f"⚠️ 用户 token 无效: {e}")
        return AuthUser(user_id=None, openid=None, raw_user_id=None, valid=False)

    openid = payload.get("openid") or payload.get("openId")
    raw_user_id = payload.get("userId") or payload.get("user_id") or payload.get("id")
    raw_user_id = str(raw_user_id) if raw_user_id is not None else None

    user_id = openid or (f"user_{raw_user_id}" if raw_user_id else None)
    return AuthUser(
        user_id=str(user_id) if user_id else None,
        openid=str(openid) if openid else None,
        raw_user_id=raw_user_id,
        valid=True,
    )
