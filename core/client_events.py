"""
client_events.py — Frontend-to-server event bus.

The frontend sends a ClientEvent to POST /event to signal state changes
during or after a query. The server acts on it (e.g. cancel an in-flight
query) without the frontend needing to know internal details.

To remove this feature entirely: delete this file and the /event endpoint
in server.py. Nothing else depends on it.

Usage (cancel flow):
    1. Frontend generates a request_id (UUID) before sending /query.
    2. POST /query with {query: "...", request_id: "<uuid>"}
    3. If the user hits cancel: POST /event with {event: "cancel", request_id: "<uuid>"}
    4. The running generate_jql coroutine detects cancellation and raises asyncio.CancelledError.

Race condition handled by CancelToken:
    The cancel event may arrive before the query handler has created its asyncio.Task.
    CancelToken is registered immediately on /query arrival (no task needed yet).
    Once the task is created it is attached to the token. If cancel already fired,
    the token cancels the task immediately on attach.
"""

import asyncio
import logging
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ClientEventType(str, Enum):
    CANCEL    = "cancel"     # user aborted the in-flight query
    HEARTBEAT = "heartbeat"  # frontend keep-alive (no server action required)


class ClientEvent(BaseModel):
    """Event sent by the frontend to notify the server of a state change.

    Fields:
        event:      What happened (see ClientEventType).
        request_id: Ties the event to a specific in-flight /query request.
    """
    event:      ClientEventType
    request_id: str


class EventAck(BaseModel):
    """Acknowledgement returned by POST /event."""
    request_id: str
    accepted:   bool
    detail:     str = ""


# ---------------------------------------------------------------------------
# CancelToken — decouples registration from task creation.
#
# Register the token the moment /query arrives (before any async work).
# Attach the asyncio.Task once it is created.
# If cancel() is called before attach(), the flag is set and the task is
# cancelled immediately when attach() is called.
# ---------------------------------------------------------------------------

class CancelToken:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self._task: asyncio.Task | None = None
        self._cancelled: bool = False

    def attach(self, task: asyncio.Task) -> None:
        """Bind the running task to this token. Cancels immediately if already cancelled."""
        self._task = task
        if self._cancelled:
            logger.info("[events] cancel was early — cancelling on attach: request_id=%s", self.request_id)
            task.cancel()

    def cancel(self) -> None:
        """Request cancellation. Works whether or not a task is attached yet."""
        self._cancelled = True
        if self._task:
            self._task.cancel()


# ---------------------------------------------------------------------------
# Registry — maps request_id → CancelToken.
# ---------------------------------------------------------------------------

_active: dict[str, CancelToken] = {}


def register(request_id: str) -> CancelToken:
    """Register a new CancelToken for request_id immediately on /query arrival.

    Call this before creating the asyncio.Task so the cancel event can always
    find the token, even if it arrives before the task exists.

    Returns the token so the caller can attach the task once created.
    """
    token = CancelToken(request_id)
    _active[request_id] = token
    logger.info("[events] registered request_id=%s (active=%d)", request_id, len(_active))
    return token


def unregister(request_id: str) -> None:
    """Remove a completed or cancelled token from the registry."""
    _active.pop(request_id, None)
    logger.info("[events] unregistered request_id=%s (active=%d)", request_id, len(_active))


def cancel(request_id: str) -> bool:
    """Cancel the query registered under request_id.

    Returns True if the token was found (task may or may not have started yet),
    False if no token exists for that request_id.
    """
    token = _active.get(request_id)
    if token is None:
        logger.warning("[events] cancel: no active query for request_id=%s", request_id)
        return False
    token.cancel()
    logger.info("[events] cancelled request_id=%s", request_id)
    return True
