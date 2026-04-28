from __future__ import annotations

import logging
from enum import Enum

from fastapi import Header
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class JiraAuthType(str, Enum):
    cloud = "cloud"
    server = "server"


class JiraCredential(BaseModel):
    """Resolved, immutable auth credential — attach directly to an httpx call."""

    model_config = {"frozen": True}

    auth: tuple[str, str] | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    @property
    def is_authenticated(self) -> bool:
        return bool(self.auth or self.headers.get("Authorization"))


class JiraProfile(BaseModel):
    """Validated Jira connection profile loaded from config/profiles.json."""

    model_config = {"extra": "ignore"}

    name: str = "default"
    jira_url: str
    jira_type: JiraAuthType = JiraAuthType.cloud
    email: str = ""
    token: str = ""

    @field_validator("jira_url")
    @classmethod
    def _normalize_url(cls, v: str) -> str:
        if not v:
            raise ValueError("jira_url must not be empty")
        return v.rstrip("/")

    @model_validator(mode="after")
    def _validate_cloud_email(self) -> "JiraProfile":
        if self.jira_type == JiraAuthType.cloud and self.token and not self.email:
            raise ValueError(
                "Jira Cloud authentication requires an email address when a token is provided"
            )
        return self

    def resolve_auth(self, token_override: str | None = None) -> JiraCredential:
        """Build the httpx auth credential for this profile.

        token_override takes precedence — used for per-request PATs from X-Jira-Token.
        Falls back to the profile-configured token, then unauthenticated.
        """
        token = token_override or self.token
        if self.jira_type == JiraAuthType.server and token:
            return JiraCredential(headers={"Authorization": f"Bearer {token}"})
        if self.email and token:
            return JiraCredential(auth=(self.email, token))
        logger.debug("No Jira credentials resolved — request will be unauthenticated")
        return JiraCredential()


async def jira_token_dep(
    x_jira_token: str | None = Header(default=None),
) -> str | None:
    """FastAPI dependency: extract X-Jira-Token from the incoming request.

    Inject via Depends(jira_token_dep) in any route handler that calls Jira.
    Returns the token string, or None when the header is absent.
    """
    return x_jira_token


async def jira_url_dep(
    x_jira_url: str | None = Header(default=None),
) -> str | None:
    """FastAPI dependency: extract X-Jira-Url from the incoming request.

    Inject via Depends(jira_url_dep) in any route handler that calls Jira.
    When present, overrides the profile-configured jira_url for this request.
    Returns the URL string, or None when the header is absent.
    """
    return x_jira_url
