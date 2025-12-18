"""Web server for the Midnight Renegade art walk prototype.

The server fulfills three responsibilities:
1. Acts as the authoritative game-state manager, tracking mobile participants,
   the volumetric display location, and the derived state (mean distance + color).
2. Serves the companion mobile web app and the password-protected admin portal.
3. Streams state updates over WebSockets to both mobile clients and the volumetric
   display driver (via a dedicated /ws/display channel).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Tuple

from fastapi import (
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.sessions import SessionMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from midnight_walk.database import (
    BeaconFindModel,
    BeaconModel,
    DisplayLocationModel,
    FakePlayerModel,
    MinigameModel,
    SpiritModel,
    StoryEventModel,
    create_async_engine_from_url,
    get_database_url,
    get_db_session,
    init_database,
    set_db_engine,
)

try:
    from bazel_tools.tools.python.runfiles import runfiles
except ImportError:
    # Fallback for non-Bazel environments (e.g., direct Python execution)
    runfiles = None

# Default location is the Conservatory of Flowers in Golden Gate Park.
DEFAULT_DISPLAY_LOCATION = (37.7726, -122.4607)

ADMIN_USERNAME = os.getenv("MIDNIGHT_ADMIN_USER", "curator")
ADMIN_PASSWORD = os.getenv("MIDNIGHT_ADMIN_PASSWORD", "midnight")
ADMIN_SESSION_SECRET = os.getenv("MIDNIGHT_ADMIN_SESSION_SECRET", "dev-secret")
ADMIN_REALM = "Midnight Walk Admin"

BEACON_STATES = ("undiscovered", "discovered", "encounter", "vanquished")

STORY_EVENT_TEXT = {
    "discovered": "{name} HAS BEEN DISCOVERED!",
    "encounter": "AN ENCOUNTER HAS BEGUN WITH {name} AT THE RIFT GATE.",
    "vanquished": "{name} HAS BEEN VANQUISHED.",
}


def _format_story_message(spirit_name: str | None, state: str) -> str | None:
    template = STORY_EVENT_TEXT.get(state)
    if not template:
        return None
    name = (spirit_name or "AN UNKNOWN SPIRIT").upper()
    return template.format(name=name)


security = HTTPBasic()

logger = logging.getLogger(__name__)


def _get_templates_dir() -> Path:
    """Get the templates directory using Bazel runfiles or fallback to local path."""
    if runfiles is not None:
        r = runfiles.Create()
        if r:
            # Try the workspace-relative path first
            template_path = r.Rlocation("volumetric-display/midnight_walk/templates")
            if template_path and Path(template_path).exists():
                return Path(template_path)
            # Also try without workspace prefix (for some Bazel setups)
            template_path = r.Rlocation("midnight_walk/templates")
            if template_path and Path(template_path).exists():
                return Path(template_path)
    # Fallback: assume we're running from the repo root or templates are in the same dir
    # Try relative to this file's location
    script_dir = Path(__file__).parent
    templates_dir = script_dir / "templates"
    if templates_dir.exists():
        return templates_dir
    # Last resort: try from current working directory
    return Path("midnight_walk/templates")


def _init_jinja_env() -> Environment:
    """Initialize Jinja2 environment with template loader."""
    templates_dir = _get_templates_dir()
    logger.info(f"Loading templates from: {templates_dir}")
    return Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=True)


@dataclass
class Beacon:
    id: str
    latitude: float
    longitude: float
    search_radius_meters: float
    spirit_id: str | None = None
    state: str = "undiscovered"


class DisplayLocationPayload(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    image_url: str | None = Field(None, max_length=500)


class BeaconPayload(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    search_radius_meters: float = Field(..., gt=0.0, le=10000.0)


class BeaconUpdatePayload(BaseModel):
    spirit_id: str | None = None
    state: str | None = None

    @field_validator("state")
    def validate_state(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value not in BEACON_STATES:
            raise ValueError(f"State must be one of {BEACON_STATES}")
        return value


class SpiritPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    blurb: str = Field(..., min_length=1)
    image_url: str | None = Field(None, max_length=500)
    totem_image_url: str | None = Field(None, max_length=500)
    current_activity: str | None = Field(None)


class BeaconStatePayload(BaseModel):
    state: str = Field(..., description="New state for the beacon.")

    @field_validator("state")
    def validate_state(cls, value: str) -> str:
        if value not in BEACON_STATES:
            raise ValueError(f"State must be one of {BEACON_STATES}")
        return value


class BeaconImportRecord(BaseModel):
    id: str
    lat: float
    lon: float
    search_radius_meters: float
    spirit_id: str | None = None
    state: str = "undiscovered"

    @field_validator("state")
    def validate_import_state(cls, value: str) -> str:
        if value not in BEACON_STATES:
            raise ValueError(f"State must be one of {BEACON_STATES}")
        return value


class BeaconImportRequest(BaseModel):
    beacons: List[BeaconImportRecord]


class SpiritImportRecord(BaseModel):
    id: str
    name: str
    blurb: str
    image_url: str | None = None
    totem_image_url: str | None = None
    current_activity: str | None = None


class MinigameCompletionPayload(BaseModel):
    score: int = Field(..., ge=0)
    metadata: dict[str, object] | None = None


class MinigameCreatePayload(BaseModel):
    spirit_id: str
    html_content: str = Field(..., min_length=1)


class SpiritImportRequest(BaseModel):
    spirits: List[SpiritImportRecord]


class MidnightGameState:
    def __init__(
        self, default_location: Tuple[float, float], db_session: AsyncSession | None = None
    ):
        self._lock = asyncio.Lock()
        self._display_lat, self._display_lon = default_location
        self._display_image_url: str | None = None
        self._beacons: Dict[str, Beacon] = {}
        self._db_session = db_session
        self._active_encounter_spirit_id: str | None = None  # Track active encounter

    async def set_display_location(
        self,
        lat: float,
        lon: float,
        db_session: AsyncSession | None = None,
        image_url: str | None = None,
    ) -> None:
        async with self._lock:
            self._display_lat = lat
            self._display_lon = lon
            if image_url is not None:
                self._display_image_url = image_url
        # Persist to database
        session = db_session or self._db_session
        if session:
            result = await session.execute(
                select(DisplayLocationModel).where(DisplayLocationModel.id == "rift_gate")
            )
            location_model = result.scalar_one_or_none()
            if location_model:
                location_model.latitude = lat
                location_model.longitude = lon
                if image_url is not None:
                    location_model.image_url = image_url
            else:
                location_model = DisplayLocationModel(
                    id="rift_gate", latitude=lat, longitude=lon, image_url=image_url
                )
                session.add(location_model)
            await session.commit()

    async def get_display_location(self) -> Tuple[float, float, str | None]:
        async with self._lock:
            return self._display_lat, self._display_lon, self._display_image_url

    async def load_display_location_from_db(self, db_session: AsyncSession) -> None:
        """Load display location from database into memory."""
        result = await db_session.execute(
            select(DisplayLocationModel).where(DisplayLocationModel.id == "rift_gate")
        )
        location_model = result.scalar_one_or_none()
        if location_model:
            async with self._lock:
                self._display_lat = location_model.latitude
                self._display_lon = location_model.longitude
                self._display_image_url = location_model.image_url
        else:
            # Initialize with default if not in DB
            async with self._lock:
                default_lat, default_lon = DEFAULT_DISPLAY_LOCATION
                self._display_lat = default_lat
                self._display_lon = default_lon
                self._display_image_url = None
            # Save default to DB
            location_model = DisplayLocationModel(
                id="rift_gate", latitude=default_lat, longitude=default_lon, image_url=None
            )
            db_session.add(location_model)
            await db_session.commit()

    async def add_beacon(
        self,
        beacon_id: str,
        lat: float,
        lon: float,
        search_radius_meters: float,
        db_session: AsyncSession | None = None,
        spirit_id: str | None = None,
        state: str = "undiscovered",
    ) -> None:
        async with self._lock:
            self._beacons[beacon_id] = Beacon(
                beacon_id, lat, lon, search_radius_meters, spirit_id, state
            )
            # Persist to database
            session = db_session or self._db_session
            if session:
                beacon_model = BeaconModel(
                    id=beacon_id,
                    latitude=lat,
                    longitude=lon,
                    search_radius_meters=search_radius_meters,
                    spirit_id=spirit_id,
                    state=state,
                )
                session.add(beacon_model)
                await session.commit()

    async def update_beacon(
        self,
        beacon_id: str,
        lat: float,
        lon: float,
        search_radius_meters: float,
        db_session: AsyncSession | None = None,
        spirit_id: str | None = None,
        state: str | None = None,
    ) -> None:
        async with self._lock:
            existing = self._beacons.get(beacon_id)
            if not existing:
                return
            resolved_spirit = spirit_id if spirit_id is not None else existing.spirit_id
            resolved_state = state if state is not None else existing.state
            self._beacons[beacon_id] = Beacon(
                beacon_id, lat, lon, search_radius_meters, resolved_spirit, resolved_state
            )
            # Update in database
            session = db_session or self._db_session
            if session:
                result = await session.execute(
                    select(BeaconModel).where(BeaconModel.id == beacon_id)
                )
                beacon_model = result.scalar_one_or_none()
                if beacon_model:
                    beacon_model.latitude = lat
                    beacon_model.longitude = lon
                    beacon_model.search_radius_meters = search_radius_meters
                    if spirit_id is not None:
                        beacon_model.spirit_id = spirit_id
                    if state is not None:
                        beacon_model.state = state
                    await session.commit()

    async def set_beacon_state(
        self,
        beacon_id: str,
        new_state: str,
        db_session: AsyncSession | None = None,
    ) -> bool:
        if new_state not in BEACON_STATES:
            raise ValueError("Invalid beacon state")
        async with self._lock:
            beacon = self._beacons.get(beacon_id)
            if not beacon:
                return False
            beacon.state = new_state
        session = db_session or self._db_session
        if session:
            result = await session.execute(select(BeaconModel).where(BeaconModel.id == beacon_id))
            beacon_model = result.scalar_one_or_none()
            if not beacon_model:
                return False
            beacon_model.state = new_state
            await session.commit()
        return True

    async def remove_beacon(self, beacon_id: str, db_session: AsyncSession | None = None) -> None:
        async with self._lock:
            self._beacons.pop(beacon_id, None)
            # Remove from database
            session = db_session or self._db_session
            if session:
                result = await session.execute(
                    select(BeaconModel).where(BeaconModel.id == beacon_id)
                )
                beacon_model = result.scalar_one_or_none()
                if beacon_model:
                    await session.delete(beacon_model)
                    await session.commit()

    async def get_beacons(self, db_session: AsyncSession | None = None) -> Dict[str, Beacon]:
        async with self._lock:
            # Load from database if available
            session = db_session or self._db_session
            if session:
                result = await session.execute(select(BeaconModel))
                beacon_models = result.scalars().all()
                # Update in-memory cache
                self._beacons = {
                    model.id: Beacon(
                        model.id,
                        model.latitude,
                        model.longitude,
                        model.search_radius_meters,
                        model.spirit_id,
                        getattr(model, "state", "undiscovered"),
                    )
                    for model in beacon_models
                }
            return self._beacons.copy()

    async def load_beacons_from_db(self, db_session: AsyncSession) -> None:
        """Load all beacons from the database into memory."""
        async with self._lock:
            result = await db_session.execute(select(BeaconModel))
            beacon_models = result.scalars().all()
            self._beacons = {
                model.id: Beacon(
                    model.id,
                    model.latitude,
                    model.longitude,
                    model.search_radius_meters,
                    model.spirit_id,
                    getattr(model, "state", "undiscovered"),
                )
                for model in beacon_models
            }

    async def snapshot(self) -> Dict[str, object]:
        async with self._lock:
            return {
                "serverTime": time.time(),
                "displayLocation": {
                    "lat": self._display_lat,
                    "lon": self._display_lon,
                    "imageUrl": self._display_image_url,
                },
                "activeEncounter": (
                    {
                        "spiritId": self._active_encounter_spirit_id,
                    }
                    if self._active_encounter_spirit_id
                    else None
                ),
            }


class ConnectionRegistry:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._mobile: Dict[str, WebSocket] = {}
        self._displays: Dict[str, WebSocket] = {}

    async def register_mobile(self, client_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._mobile[client_id] = websocket

    async def unregister_mobile(self, client_id: str) -> None:
        async with self._lock:
            self._mobile.pop(client_id, None)

    async def register_display(self, display_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._displays[display_id] = websocket

    async def unregister_display(self, display_id: str) -> None:
        async with self._lock:
            self._displays.pop(display_id, None)

    async def broadcast_to_mobiles(self, payload: Dict[str, object]) -> None:
        await self._broadcast(self._mobile, payload)

    async def broadcast_to_displays(self, payload: Dict[str, object]) -> None:
        await self._broadcast(self._displays, payload)

    async def _broadcast(
        self,
        recipients: Dict[str, WebSocket],
        payload: Dict[str, object],
    ) -> None:
        stale = []
        for client_id, websocket in recipients.items():
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(client_id)
        for client_id in stale:
            recipients.pop(client_id, None)


# Template strings removed - now using Jinja2 templates from midnight_walk/templates/


def create_app() -> FastAPI:
    # Initialize state and registry before creating the app
    state = MidnightGameState(DEFAULT_DISPLAY_LOCATION)
    registry = ConnectionRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Lifespan context manager for database initialization and cleanup."""
        # Startup
        database_url = get_database_url()
        engine = create_async_engine_from_url(database_url)
        set_db_engine(engine)  # Set global engine for get_db_session dependency

        await init_database(engine)
        # Load beacons from database into state
        from sqlalchemy.ext.asyncio import async_sessionmaker

        async_session_maker = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        async with async_session_maker() as session:
            await state.load_beacons_from_db(session)
            await state.load_display_location_from_db(session)

        yield

        # Shutdown
        await engine.dispose()

    app = FastAPI(title="Midnight Renegade Server", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
    # Only enable HTTPS redirect if FORCE_HTTPS is set AND we're not on Heroku
    # (Heroku handles HTTPS termination at the router level)
    if os.getenv("FORCE_HTTPS", "0") == "1" and not os.getenv("DYNO"):
        app.add_middleware(HTTPSRedirectMiddleware)
    app.add_middleware(SessionMiddleware, secret_key=ADMIN_SESSION_SECRET, https_only=False)

    # Initialize Jinja2 template environment
    jinja_env = _init_jinja_env()

    async def push_state():
        snapshot = await state.snapshot()
        await registry.broadcast_to_displays({"type": "display_state", **snapshot})
        await registry.broadcast_to_mobiles({"type": "state_update", **snapshot})

    async def _record_story_event(beacon_id: str, new_state: str, db: AsyncSession) -> None:
        message = None
        spirit_id = None
        spirit_name = None
        result = await db.execute(
            select(BeaconModel, SpiritModel)
            .outerjoin(SpiritModel, BeaconModel.spirit_id == SpiritModel.id)
            .where(BeaconModel.id == beacon_id)
        )
        row = result.first()
        if row:
            beacon_model, spirit_model = row
            if spirit_model:
                spirit_id = spirit_model.id
                spirit_name = spirit_model.name
            message = _format_story_message(spirit_name, new_state)
        if not message:
            return
        event = StoryEventModel(
            id=str(uuid.uuid4()),
            created_at=time.time(),
            event_type=new_state,
            beacon_id=beacon_id,
            spirit_id=spirit_id,
            spirit_name=spirit_name,
            message=message,
        )
        db.add(event)
        await db.commit()

    async def _transition_beacon(
        beacon_id: str,
        new_state: str,
        db: AsyncSession,
        *,
        emit_story: bool = True,
        push_update: bool = True,
    ) -> bool:
        # Get beacon to check spirit_id before transition
        result = await db.execute(select(BeaconModel).where(BeaconModel.id == beacon_id))
        beacon_model = result.scalar_one_or_none()

        updated = await state.set_beacon_state(beacon_id, new_state, db_session=db)
        if not updated:
            return False

        # Manage active encounter tracking
        if beacon_model and beacon_model.spirit_id:
            async with state._lock:
                if new_state == "encounter":
                    # Set as active encounter
                    state._active_encounter_spirit_id = beacon_model.spirit_id
                elif (
                    new_state == "vanquished"
                    and state._active_encounter_spirit_id == beacon_model.spirit_id
                ):
                    # Clear active encounter when vanquished
                    state._active_encounter_spirit_id = None

        if emit_story:
            await _record_story_event(beacon_id, new_state, db)
        if push_update:
            await push_state()
        return True

    async def admin_guard(credentials: HTTPBasicCredentials = Depends(security)) -> None:
        correct_user = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
        correct_pass = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
        if not (correct_user and correct_pass):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": f'Basic realm="{ADMIN_REALM}"'},
            )

    @app.get("/", response_class=HTMLResponse)
    async def mobile_portal():
        template = jinja_env.get_template("mobile.html")
        return HTMLResponse(content=template.render())

    @app.get("/about", response_class=HTMLResponse)
    async def about_page():
        template = jinja_env.get_template("about.html")
        return HTMLResponse(content=template.render())

    def is_admin(request: Request) -> bool:
        return request.session.get("admin_authenticated") is True

    @app.get("/admin/login", response_class=HTMLResponse)
    async def admin_login_page(request: Request):
        if is_admin(request):
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
        template = jinja_env.get_template("admin_login.html")
        return HTMLResponse(content=template.render(error_msg=""))

    @app.post("/admin/login")
    async def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            request.session["admin_authenticated"] = True
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
        template = jinja_env.get_template("admin_login.html")
        return HTMLResponse(
            content=template.render(error_msg="Invalid credentials."),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    @app.get("/admin/logout")
    async def admin_logout(request: Request):
        request.session.clear()
        return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_portal(request: Request):
        if not is_admin(request):
            return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)
        template = jinja_env.get_template("admin.html")
        return HTMLResponse(content=template.render())

    @app.get("/admin/beacons", response_class=HTMLResponse)
    async def admin_beacons_page(request: Request):
        """Admin page for managing beacon-spirit associations."""
        if not is_admin(request):
            return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)
        template = jinja_env.get_template("admin_beacons.html")
        return HTMLResponse(content=template.render())

    @app.get("/admin/spirits", response_class=HTMLResponse)
    async def admin_spirits_page(request: Request):
        """Admin page for managing spirits."""
        if not is_admin(request):
            return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)
        template = jinja_env.get_template("admin_spirits.html")
        return HTMLResponse(content=template.render())

    @app.get("/admin/data", response_class=HTMLResponse)
    async def admin_data_page(request: Request):
        """Admin page for exporting/importing data."""
        if not is_admin(request):
            return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)
        template = jinja_env.get_template("admin_data.html")
        return HTMLResponse(content=template.render())

    @app.get("/admin/rift-gate", response_class=HTMLResponse)
    async def admin_rift_gate_page(request: Request):
        """Admin page for configuring the rift gate (display location)."""
        if not is_admin(request):
            return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)
        template = jinja_env.get_template("admin_rift_gate.html")
        return HTMLResponse(content=template.render())

    @app.get("/admin/story", response_class=HTMLResponse)
    async def admin_story_page(request: Request):
        """Admin page for managing the story log."""
        if not is_admin(request):
            return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)
        template = jinja_env.get_template("admin_story.html")
        return HTMLResponse(content=template.render())

    @app.get("/admin/minigames", response_class=HTMLResponse)
    async def admin_minigames_page(request: Request):
        """Admin page for managing minigames."""
        if not is_admin(request):
            return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)
        template = jinja_env.get_template("admin_minigames.html")
        return HTMLResponse(content=template.render())

    @app.get("/api/admin/beacons/export", dependencies=[Depends(admin_guard)])
    async def export_beacons_json(db: AsyncSession = Depends(get_db_session)):
        result = await db.execute(select(BeaconModel))
        rows = result.scalars().all()
        payload = [
            {
                "id": row.id,
                "lat": row.latitude,
                "lon": row.longitude,
                "search_radius_meters": row.search_radius_meters,
                "spirit_id": row.spirit_id,
                "state": getattr(row, "state", "undiscovered"),
            }
            for row in rows
        ]
        return JSONResponse({"beacons": payload})

    @app.post("/api/admin/beacons/import", dependencies=[Depends(admin_guard)])
    async def import_beacons_json(
        request_payload: BeaconImportRequest, db: AsyncSession = Depends(get_db_session)
    ):
        await db.execute(delete(BeaconModel))
        for record in request_payload.beacons:
            db.add(
                BeaconModel(
                    id=record.id,
                    latitude=record.lat,
                    longitude=record.lon,
                    search_radius_meters=record.search_radius_meters,
                    spirit_id=record.spirit_id,
                    state=record.state,
                )
            )
        await db.commit()
        await state.load_beacons_from_db(db)
        await push_state()
        return {"status": "ok", "imported": len(request_payload.beacons)}

    @app.get("/api/admin/spirits/export", dependencies=[Depends(admin_guard)])
    async def export_spirits_json(db: AsyncSession = Depends(get_db_session)):
        result = await db.execute(select(SpiritModel))
        rows = result.scalars().all()
        payload = [
            {
                "id": row.id,
                "name": row.name,
                "blurb": row.blurb,
                "image_url": row.image_url,
                "totem_image_url": row.totem_image_url,
                "current_activity": row.current_activity,
            }
            for row in rows
        ]
        return JSONResponse({"spirits": payload})

    @app.post("/api/admin/spirits/import", dependencies=[Depends(admin_guard)])
    async def import_spirits_json(
        request_payload: SpiritImportRequest, db: AsyncSession = Depends(get_db_session)
    ):
        await db.execute(delete(SpiritModel))
        for record in request_payload.spirits:
            db.add(
                SpiritModel(
                    id=record.id,
                    name=record.name,
                    blurb=record.blurb,
                    image_url=record.image_url,
                    totem_image_url=record.totem_image_url,
                    current_activity=record.current_activity,
                )
            )
        await db.commit()
        return {"status": "ok", "imported": len(request_payload.spirits)}

    @app.get("/admin/nfc/{beacon_id}", response_class=HTMLResponse)
    async def admin_nfc_writer(
        beacon_id: str,
        request: Request,
        db: AsyncSession = Depends(get_db_session),
    ):
        """Internal NFC writer page that uses Web NFC to program tags."""
        if not is_admin(request):
            return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)
        result = await db.execute(select(BeaconModel).where(BeaconModel.id == beacon_id))
        beacon_model = result.scalar_one_or_none()
        if not beacon_model:
            raise HTTPException(status_code=404, detail="Beacon not found")
        nfc_url = f"https://midnight.fughil.li/found?id={beacon_model.id}"
        template = jinja_env.get_template("admin_nfc_writer.html")
        return HTMLResponse(
            content=template.render(
                beacon_id=beacon_model.id,
                nfc_url=nfc_url,
            )
        )

    @app.get("/api/display-location")
    async def get_display_location():
        lat, lon, image_url = await state.get_display_location()
        return {"lat": lat, "lon": lon, "image_url": image_url}

    @app.post("/api/display-location", dependencies=[Depends(admin_guard)])
    async def set_display_location(
        payload: DisplayLocationPayload, db: AsyncSession = Depends(get_db_session)
    ):
        await state.set_display_location(
            payload.lat, payload.lon, db_session=db, image_url=payload.image_url
        )
        await push_state()
        return {"status": "ok"}

    @app.get("/api/beacons", dependencies=[Depends(admin_guard)])
    async def get_beacons(db: AsyncSession = Depends(get_db_session)):
        beacons = await state.get_beacons(db_session=db)
        return {
            "beacons": [
                {
                    "id": b.id,
                    "lat": b.latitude,
                    "lon": b.longitude,
                    "searchRadiusMeters": b.search_radius_meters,
                    "spiritId": b.spirit_id,
                    "state": b.state,
                }
                for b in beacons.values()
            ]
        }

    @app.post("/api/beacons", dependencies=[Depends(admin_guard)])
    async def create_beacon(payload: BeaconPayload, db: AsyncSession = Depends(get_db_session)):
        beacon_id = str(uuid.uuid4())
        await state.add_beacon(
            beacon_id,
            payload.lat,
            payload.lon,
            payload.search_radius_meters,
            db_session=db,
            state="undiscovered",
        )
        await push_state()
        return {"id": beacon_id, "status": "ok"}

    @app.put("/api/beacons/{beacon_id}", dependencies=[Depends(admin_guard)])
    async def update_beacon(
        beacon_id: str, payload: BeaconPayload, db: AsyncSession = Depends(get_db_session)
    ):
        await state.update_beacon(
            beacon_id, payload.lat, payload.lon, payload.search_radius_meters, db_session=db
        )
        await push_state()
        return {"status": "ok"}

    @app.delete("/api/beacons/{beacon_id}", dependencies=[Depends(admin_guard)])
    async def delete_beacon(beacon_id: str, db: AsyncSession = Depends(get_db_session)):
        await state.remove_beacon(beacon_id, db_session=db)
        await push_state()
        return {"status": "ok"}

    @app.patch("/api/beacons/{beacon_id}", dependencies=[Depends(admin_guard)])
    async def update_beacon_patch(
        beacon_id: str, payload: BeaconUpdatePayload, db: AsyncSession = Depends(get_db_session)
    ):
        """Update beacon metadata (spirit association and/or state)."""
        result = await db.execute(select(BeaconModel).where(BeaconModel.id == beacon_id))
        beacon_model = result.scalar_one_or_none()
        if not beacon_model:
            raise HTTPException(status_code=404, detail="Beacon not found")

        updated_metadata = False
        state_changed = False
        if payload.spirit_id is not None:
            beacon_model.spirit_id = payload.spirit_id
            updated_metadata = True

        if payload.state is not None:
            if payload.state not in BEACON_STATES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid state '{payload.state}'",
                )
            state_changed = await _transition_beacon(
                beacon_id, payload.state, db, push_update=False
            )

        if payload.spirit_id is not None:
            await db.commit()
            async with state._lock:
                if beacon_id in state._beacons:
                    beacon = state._beacons[beacon_id]
                    state._beacons[beacon_id] = Beacon(
                        beacon_id,
                        beacon_model.latitude,
                        beacon_model.longitude,
                        beacon_model.search_radius_meters,
                        beacon_model.spirit_id,
                        beacon.state,
                    )

        if updated_metadata or state_changed:
            await push_state()
        return {"status": "ok", "updated": updated_metadata or state_changed}

    async def _apply_state_transition(
        beacon_id: str,
        new_state: str,
        db: AsyncSession,
    ):
        updated = await _transition_beacon(beacon_id, new_state, db)
        if not updated:
            raise HTTPException(status_code=404, detail="Beacon not found")
        return {"status": "ok", "state": new_state}

    @app.post("/api/beacons/{beacon_id}/state", dependencies=[Depends(admin_guard)])
    async def admin_set_beacon_state(
        beacon_id: str, payload: BeaconStatePayload, db: AsyncSession = Depends(get_db_session)
    ):
        return await _apply_state_transition(beacon_id, payload.state, db)

    @app.post("/beacon/{beacon_id}/discovered")
    async def mark_beacon_discovered(beacon_id: str, db: AsyncSession = Depends(get_db_session)):
        return await _apply_state_transition(beacon_id, "discovered", db)

    @app.post("/beacon/{beacon_id}/returned")
    async def mark_beacon_returned(beacon_id: str, db: AsyncSession = Depends(get_db_session)):
        # Legacy endpoint - now uses "encounter" state
        return await _apply_state_transition(beacon_id, "encounter", db)

    @app.post("/beacon/{beacon_id}/vanquished")
    async def mark_beacon_vanquished(beacon_id: str, db: AsyncSession = Depends(get_db_session)):
        return await _apply_state_transition(beacon_id, "vanquished", db)

    @app.get("/api/spirits", dependencies=[Depends(admin_guard)])
    async def get_spirits(db: AsyncSession = Depends(get_db_session)):
        """Get all spirits."""
        result = await db.execute(select(SpiritModel))
        spirits = result.scalars().all()
        return {
            "spirits": [
                {
                    "id": s.id,
                    "name": s.name,
                    "blurb": s.blurb,
                    "imageUrl": s.image_url,
                    "totemImageUrl": s.totem_image_url,
                    "currentActivity": s.current_activity,
                }
                for s in spirits
            ]
        }

    @app.post("/api/spirits", dependencies=[Depends(admin_guard)])
    async def create_spirit(payload: SpiritPayload, db: AsyncSession = Depends(get_db_session)):
        """Create a new spirit."""
        spirit_id = str(uuid.uuid4())
        spirit_model = SpiritModel(
            id=spirit_id,
            name=payload.name,
            blurb=payload.blurb,
            image_url=payload.image_url,
            totem_image_url=payload.totem_image_url,
            current_activity=payload.current_activity,
        )
        db.add(spirit_model)
        await db.commit()
        return {"id": spirit_id, "status": "ok"}

    @app.put("/api/spirits/{spirit_id}", dependencies=[Depends(admin_guard)])
    async def update_spirit(
        spirit_id: str, payload: SpiritPayload, db: AsyncSession = Depends(get_db_session)
    ):
        """Update a spirit."""
        result = await db.execute(select(SpiritModel).where(SpiritModel.id == spirit_id))
        spirit_model = result.scalar_one_or_none()
        if not spirit_model:
            raise HTTPException(status_code=404, detail="Spirit not found")
        spirit_model.name = payload.name
        spirit_model.blurb = payload.blurb
        spirit_model.image_url = payload.image_url
        spirit_model.totem_image_url = payload.totem_image_url
        spirit_model.current_activity = payload.current_activity
        await db.commit()
        return {"status": "ok"}

    @app.get("/api/story")
    async def get_story_events(limit: int = 40, db: AsyncSession = Depends(get_db_session)):
        """Public story log feed."""
        clamped_limit = max(1, min(limit, 100))
        result = await db.execute(
            select(StoryEventModel).order_by(StoryEventModel.created_at.desc()).limit(clamped_limit)
        )
        events = []
        for event in result.scalars():
            events.append(
                {
                    "id": event.id,
                    "message": event.message,
                    "spiritName": event.spirit_name,
                    "spiritId": event.spirit_id,
                    "eventType": event.event_type,
                    "createdAt": datetime.fromtimestamp(event.created_at, timezone.utc).isoformat(),
                }
            )
        return {"events": events}

    @app.get("/api/admin/story", dependencies=[Depends(admin_guard)])
    async def get_story_events_admin(limit: int = 100, db: AsyncSession = Depends(get_db_session)):
        """Admin story log feed with more events."""
        clamped_limit = max(1, min(limit, 500))
        result = await db.execute(
            select(StoryEventModel).order_by(StoryEventModel.created_at.desc()).limit(clamped_limit)
        )
        events = []
        for event in result.scalars():
            events.append(
                {
                    "id": event.id,
                    "message": event.message,
                    "spiritName": event.spirit_name,
                    "spiritId": event.spirit_id,
                    "eventType": event.event_type,
                    "createdAt": datetime.fromtimestamp(event.created_at, timezone.utc).isoformat(),
                }
            )
        return {"events": events}

    @app.delete("/api/admin/story/{event_id}", dependencies=[Depends(admin_guard)])
    async def delete_story_event(event_id: str, db: AsyncSession = Depends(get_db_session)):
        """Delete a specific story event."""
        result = await db.execute(select(StoryEventModel).where(StoryEventModel.id == event_id))
        event = result.scalar_one_or_none()
        if not event:
            raise HTTPException(status_code=404, detail="Story event not found")
        await db.delete(event)
        await db.commit()
        return {"status": "ok"}

    @app.delete("/api/admin/story", dependencies=[Depends(admin_guard)])
    async def clear_story_log(db: AsyncSession = Depends(get_db_session)):
        """Clear all story events."""
        await db.execute(delete(StoryEventModel))
        await db.commit()
        return {"status": "ok", "deleted": "all"}

    @app.get("/api/public-map")
    async def get_public_map(db: AsyncSession = Depends(get_db_session)):
        """Public data set for rendering the ritual map."""
        beacon_rows = await db.execute(
            select(BeaconModel, SpiritModel).outerjoin(
                SpiritModel, BeaconModel.spirit_id == SpiritModel.id
            )
        )
        display_lat, display_lon, display_image_url = await state.get_display_location()
        beacons_payload = []
        for beacon_model, spirit_model in beacon_rows.all():
            beacons_payload.append(
                {
                    "id": beacon_model.id,
                    "lat": beacon_model.latitude,
                    "lon": beacon_model.longitude,
                    "searchRadius": beacon_model.search_radius_meters,
                    "state": getattr(beacon_model, "state", "undiscovered"),
                    "spirit": (
                        {
                            "id": spirit_model.id,
                            "name": spirit_model.name,
                            "imageUrl": spirit_model.image_url,
                            "totemImageUrl": spirit_model.totem_image_url,
                        }
                        if spirit_model
                        else None
                    ),
                }
            )
        return {
            "display": {"lat": display_lat, "lon": display_lon, "imageUrl": display_image_url},
            "beacons": beacons_payload,
        }

    @app.get("/api/nfc-lookup")
    async def lookup_spirit_by_beacon(
        beacon_uid: str,
        start_encounter: bool = False,
        db: AsyncSession = Depends(get_db_session),
    ):
        """Look up spirit UUID associated with a beacon (NFC tag) UUID.

        Args:
            beacon_uid: The NFC tag UID (beacon ID)
            start_encounter: If True, mark the spirit as being in encounter state
        """
        # Look up beacon by ID
        result = await db.execute(select(BeaconModel).where(BeaconModel.id == beacon_uid))
        beacon_model = result.scalar_one_or_none()

        if not beacon_model:
            raise HTTPException(status_code=404, detail="Beacon not found")

        # Check if there's an active encounter
        async with state._lock:
            active_encounter = state._active_encounter_spirit_id

        if active_encounter and beacon_model.spirit_id != active_encounter:
            # Another encounter is active
            return {
                "spirit_id": None,
                "beacon_id": beacon_uid,
                "error": "active_encounter",
                "message": "Cannot start encounter: another spirit is currently being confronted.",
                "active_spirit_id": active_encounter,
            }

        if not beacon_model.spirit_id:
            return {
                "spirit_id": None,
                "beacon_id": beacon_uid,
                "error": "no_spirit",
                "message": "This beacon is not associated with a spirit.",
            }

        # Get spirit details
        spirit_result = await db.execute(
            select(SpiritModel).where(SpiritModel.id == beacon_model.spirit_id)
        )
        spirit_model = spirit_result.scalar_one_or_none()

        if not spirit_model:
            return {
                "spirit_id": None,
                "beacon_id": beacon_uid,
                "error": "spirit_not_found",
                "message": "Spirit associated with beacon not found.",
            }

        response = {
            "spirit_id": spirit_model.id,
            "spirit_name": spirit_model.name,
            "beacon_id": beacon_uid,
            "beacon_state": beacon_model.state,
        }

        # If start_encounter is True, transition beacon to encounter state
        if start_encounter:
            if beacon_model.state == "encounter":
                # Already in encounter
                response["message"] = "Spirit is already in encounter state."
            else:
                # Transition to encounter state
                await _transition_beacon(beacon_uid, "encounter", db, push_update=True)
                async with state._lock:
                    state._active_encounter_spirit_id = spirit_model.id

                # Check if spirit has a minigame
                minigame_result = await db.execute(
                    select(MinigameModel).where(MinigameModel.spirit_id == spirit_model.id)
                )
                minigame = minigame_result.scalar_one_or_none()

                # Broadcast encounter event to all mobile clients
                await registry.broadcast_to_mobiles(
                    {
                        "type": "encounter_started",
                        "spirit_id": spirit_model.id,
                        "spirit_name": spirit_model.name,
                        "has_minigame": minigame is not None,
                    }
                )

                response["message"] = f"Encounter started with {spirit_model.name}."
                response["beacon_state"] = "encounter"

        return response

    @app.post("/api/spirits/{spirit_id}/vanquished")
    async def mark_spirit_vanquished(
        spirit_id: str,
        db: AsyncSession = Depends(get_db_session),
    ):
        """Mark a spirit as vanquished. This should be called after the defeat animation completes."""
        # Find all beacons associated with this spirit
        result = await db.execute(select(BeaconModel).where(BeaconModel.spirit_id == spirit_id))
        beacons = result.scalars().all()

        if not beacons:
            raise HTTPException(status_code=404, detail=f"No beacons found for spirit {spirit_id}")

        # Transition all beacons to vanquished state
        for beacon in beacons:
            await _transition_beacon(beacon.id, "vanquished", db, push_update=False)

        # Clear active encounter
        async with state._lock:
            state._active_encounter_spirit_id = None

        await push_state()

        return {
            "status": "ok",
            "spirit_id": spirit_id,
            "beacons_vanquished": [b.id for b in beacons],
        }

    @app.get("/api/encounter/active")
    async def get_active_encounter(db: AsyncSession = Depends(get_db_session)):
        """Get the currently active encounter, if any."""
        async with state._lock:
            active_spirit_id = state._active_encounter_spirit_id

        if not active_spirit_id:
            return {"active": False}

        # Get spirit details
        result = await db.execute(select(SpiritModel).where(SpiritModel.id == active_spirit_id))
        spirit_model = result.scalar_one_or_none()

        if not spirit_model:
            # Clear invalid encounter
            async with state._lock:
                state._active_encounter_spirit_id = None
            return {"active": False}

        # Check if spirit has a minigame
        minigame_result = await db.execute(
            select(MinigameModel).where(MinigameModel.spirit_id == active_spirit_id)
        )
        minigame = minigame_result.scalar_one_or_none()

        return {
            "active": True,
            "spirit_id": spirit_model.id,
            "spirit_name": spirit_model.name,
            "has_minigame": minigame is not None,
        }

    @app.get("/api/minigames/{spirit_id}")
    async def get_minigame(spirit_id: str, db: AsyncSession = Depends(get_db_session)):
        """Get the minigame HTML for a spirit."""
        result = await db.execute(select(MinigameModel).where(MinigameModel.spirit_id == spirit_id))
        minigame = result.scalar_one_or_none()

        if not minigame:
            raise HTTPException(status_code=404, detail="Minigame not found for this spirit")

        # Inject spirit ID into the HTML so the minigame can report completion
        html_content = minigame.html_content.replace("spiritId: null,", f'spiritId: "{spirit_id}",')

        return HTMLResponse(content=html_content, media_type="text/html")

    @app.post("/api/minigames/{spirit_id}/complete")
    async def complete_minigame(
        spirit_id: str,
        payload: MinigameCompletionPayload,
        db: AsyncSession = Depends(get_db_session),
    ):
        """Report minigame completion. This will vanquish the spirit if successful."""
        # Verify spirit exists
        result = await db.execute(select(SpiritModel).where(SpiritModel.id == spirit_id))
        spirit_model = result.scalar_one_or_none()

        if not spirit_model:
            raise HTTPException(status_code=404, detail="Spirit not found")

        # Verify this is the active encounter
        async with state._lock:
            active_spirit_id = state._active_encounter_spirit_id

        if active_spirit_id != spirit_id:
            raise HTTPException(
                status_code=400, detail="This spirit is not currently in an active encounter"
            )

        # Vanquish the spirit (find all beacons and transition them)
        beacon_result = await db.execute(
            select(BeaconModel).where(BeaconModel.spirit_id == spirit_id)
        )
        beacons = beacon_result.scalars().all()

        if beacons:
            # Transition all beacons to vanquished state
            for beacon in beacons:
                await _transition_beacon(beacon.id, "vanquished", db, push_update=False)

            # Clear active encounter
            async with state._lock:
                state._active_encounter_spirit_id = None

            await push_state()

        return {
            "status": "ok",
            "spirit_id": spirit_id,
            "spirit_name": spirit_model.name,
            "score": payload.score,
        }

    @app.get("/api/admin/minigames", dependencies=[Depends(admin_guard)])
    async def get_all_minigames(db: AsyncSession = Depends(get_db_session)):
        """Get all minigames with their associated spirits."""
        result = await db.execute(
            select(MinigameModel, SpiritModel).outerjoin(
                SpiritModel, MinigameModel.spirit_id == SpiritModel.id
            )
        )
        minigames = []
        for minigame_model, spirit_model in result.all():
            minigames.append(
                {
                    "id": minigame_model.id,
                    "spirit_id": minigame_model.spirit_id,
                    "spirit_name": spirit_model.name if spirit_model else None,
                }
            )
        return {"minigames": minigames}

    @app.get("/api/admin/minigames/{spirit_id}", dependencies=[Depends(admin_guard)])
    async def get_minigame_content(spirit_id: str, db: AsyncSession = Depends(get_db_session)):
        """Get the HTML content of a minigame."""
        result = await db.execute(select(MinigameModel).where(MinigameModel.spirit_id == spirit_id))
        minigame = result.scalar_one_or_none()
        if not minigame:
            raise HTTPException(status_code=404, detail="Minigame not found")
        return {"spirit_id": spirit_id, "html_content": minigame.html_content}

    @app.post("/api/admin/minigames", dependencies=[Depends(admin_guard)])
    async def create_minigame(
        payload: MinigameCreatePayload,
        db: AsyncSession = Depends(get_db_session),
    ):
        """Create or update a minigame for a spirit."""
        # Verify spirit exists
        result = await db.execute(select(SpiritModel).where(SpiritModel.id == payload.spirit_id))
        spirit = result.scalar_one_or_none()
        if not spirit:
            raise HTTPException(status_code=404, detail="Spirit not found")

        # Check if minigame already exists
        existing = await db.execute(
            select(MinigameModel).where(MinigameModel.spirit_id == payload.spirit_id)
        )
        minigame = existing.scalar_one_or_none()

        if minigame:
            # Update existing
            minigame.html_content = payload.html_content
        else:
            # Create new
            minigame_id = str(uuid.uuid4())
            minigame = MinigameModel(
                id=minigame_id, spirit_id=payload.spirit_id, html_content=payload.html_content
            )
            db.add(minigame)

        await db.commit()
        return {"status": "ok", "minigame_id": minigame.id, "spirit_id": payload.spirit_id}

    @app.delete("/api/admin/minigames/{spirit_id}", dependencies=[Depends(admin_guard)])
    async def delete_minigame(spirit_id: str, db: AsyncSession = Depends(get_db_session)):
        """Delete a minigame for a spirit."""
        result = await db.execute(select(MinigameModel).where(MinigameModel.spirit_id == spirit_id))
        minigame = result.scalar_one_or_none()
        if not minigame:
            raise HTTPException(status_code=404, detail="Minigame not found")

        await db.delete(minigame)
        await db.commit()
        return {"status": "ok"}

    @app.get("/api/rift/players-nearby")
    async def get_players_near_rift(
        radius_meters: float = 10.0, db: AsyncSession = Depends(get_db_session)
    ):
        """Get count of fake players within specified radius of the rift gate.

        Args:
            radius_meters: Radius in meters (default 10m for Tharok thawing)
        """
        display_lat, display_lon, _ = await state.get_display_location()

        # Get fake players from database
        fake_players_result = await db.execute(select(FakePlayerModel))
        fake_players = fake_players_result.scalars().all()

        # Count fake players within radius
        fake_nearby_count = 0
        for fake_player in fake_players:
            lat1, lon1 = math.radians(display_lat), math.radians(display_lon)
            lat2, lon2 = math.radians(fake_player.latitude), math.radians(fake_player.longitude)

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            distance_meters = 6371000 * c

            if distance_meters <= radius_meters:
                fake_nearby_count += 1

        return {
            "player_count": fake_nearby_count,
            "fake_player_count": fake_nearby_count,
            "radius_meters": radius_meters,
            "rift_location": {"lat": display_lat, "lon": display_lon},
        }

    @app.get("/api/fake-players", dependencies=[Depends(admin_guard)])
    async def get_fake_players(db: AsyncSession = Depends(get_db_session)):
        """Get all fake players."""
        result = await db.execute(select(FakePlayerModel))
        fake_players = result.scalars().all()
        return {
            "fake_players": [
                {
                    "id": fp.id,
                    "lat": fp.latitude,
                    "lon": fp.longitude,
                    "name": fp.name,
                }
                for fp in fake_players
            ]
        }

    @app.post("/api/fake-players", dependencies=[Depends(admin_guard)])
    async def create_fake_player(
        lat: float = Form(...),
        lon: float = Form(...),
        name: str | None = Form(None),
        db: AsyncSession = Depends(get_db_session),
    ):
        """Create a new fake player."""
        fake_player_id = str(uuid.uuid4())
        fake_player = FakePlayerModel(
            id=fake_player_id,
            latitude=lat,
            longitude=lon,
            name=name,
        )
        db.add(fake_player)
        await db.commit()
        return {"id": fake_player_id, "status": "ok"}

    @app.put("/api/fake-players/{player_id}", dependencies=[Depends(admin_guard)])
    async def update_fake_player(
        player_id: str,
        lat: float = Form(...),
        lon: float = Form(...),
        name: str | None = Form(None),
        db: AsyncSession = Depends(get_db_session),
    ):
        """Update a fake player's position."""
        result = await db.execute(select(FakePlayerModel).where(FakePlayerModel.id == player_id))
        fake_player = result.scalar_one_or_none()
        if not fake_player:
            raise HTTPException(status_code=404, detail="Fake player not found")
        fake_player.latitude = lat
        fake_player.longitude = lon
        fake_player.name = name
        await db.commit()
        return {"status": "ok"}

    @app.delete("/api/fake-players/{player_id}", dependencies=[Depends(admin_guard)])
    async def delete_fake_player(player_id: str, db: AsyncSession = Depends(get_db_session)):
        """Delete a fake player."""
        result = await db.execute(select(FakePlayerModel).where(FakePlayerModel.id == player_id))
        fake_player = result.scalar_one_or_none()
        if not fake_player:
            raise HTTPException(status_code=404, detail="Fake player not found")
        await db.delete(fake_player)
        await db.commit()
        return {"status": "ok"}

    @app.delete("/api/spirits/{spirit_id}", dependencies=[Depends(admin_guard)])
    async def delete_spirit(spirit_id: str, db: AsyncSession = Depends(get_db_session)):
        """Delete a spirit."""
        result = await db.execute(select(SpiritModel).where(SpiritModel.id == spirit_id))
        spirit_model = result.scalar_one_or_none()
        if not spirit_model:
            raise HTTPException(status_code=404, detail="Spirit not found")
        await db.delete(spirit_model)
        await db.commit()
        return {"status": "ok"}

    @app.get("/api/export", dependencies=[Depends(admin_guard)])
    async def export_data(db: AsyncSession = Depends(get_db_session)):
        """Export spirits and beacons as JSON."""
        beacon_result = await db.execute(select(BeaconModel))
        spirit_result = await db.execute(select(SpiritModel))
        beacons = [
            {
                "id": b.id,
                "latitude": b.latitude,
                "longitude": b.longitude,
                "search_radius_meters": b.search_radius_meters,
                "spirit_id": b.spirit_id,
                "status": b.status,
            }
            for b in beacon_result.scalars().all()
        ]
        spirits_payload = [
            {
                "id": s.id,
                "name": s.name,
                "blurb": s.blurb,
                "image_url": s.image_url,
                "totem_image_url": s.totem_image_url,
                "current_activity": s.current_activity,
            }
            for s in spirit_result.scalars().all()
        ]
        return {"beacons": beacons, "spirits": spirits_payload}

    @app.post("/api/import", dependencies=[Depends(admin_guard)])
    async def import_data(payload: Dict[str, object], db: AsyncSession = Depends(get_db_session)):
        """Replace beacon and spirit data with uploaded JSON."""
        spirits_payload = payload.get("spirits", [])
        beacons_payload = payload.get("beacons", [])
        if not isinstance(spirits_payload, list) or not isinstance(beacons_payload, list):
            raise HTTPException(status_code=400, detail="Invalid payload; expected lists.")

        await db.execute(delete(BeaconFindModel))
        await db.execute(delete(BeaconModel))
        await db.execute(delete(SpiritModel))

        for spirit in spirits_payload:
            try:
                db.add(
                    SpiritModel(
                        id=spirit["id"],
                        name=spirit["name"],
                        blurb=spirit["blurb"],
                        image_url=spirit.get("image_url"),
                        totem_image_url=spirit.get("totem_image_url"),
                        current_activity=spirit.get("current_activity"),
                    )
                )
            except KeyError as exc:
                raise HTTPException(status_code=400, detail=f"Missing spirit field {exc}") from exc

        for beacon in beacons_payload:
            try:
                status = beacon.get("status", "undiscovered")
                if status not in BEACON_STATES:
                    raise ValueError(f"Invalid status '{status}'")
                db.add(
                    BeaconModel(
                        id=beacon["id"],
                        latitude=beacon["latitude"],
                        longitude=beacon["longitude"],
                        search_radius_meters=beacon["search_radius_meters"],
                        spirit_id=beacon.get("spirit_id"),
                        status=status,
                    )
                )
            except KeyError as exc:
                raise HTTPException(status_code=400, detail=f"Missing beacon field {exc}") from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        await db.commit()
        await state.load_beacons_from_db(db)
        await push_state()
        return {"status": "ok"}

    @app.get("/found")
    async def found_beacon(request: Request, id: str, db: AsyncSession = Depends(get_db_session)):
        """Handle NFC tag scan - log the find and redirect to spirit page."""
        # Log the find
        find_id = str(uuid.uuid4())
        find_model = BeaconFindModel(
            id=find_id,
            beacon_id=id,
            found_at=time.time(),
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None,
        )
        db.add(find_model)
        await db.commit()

        # Get the beacon and its associated spirit
        result = await db.execute(select(BeaconModel).where(BeaconModel.id == id))
        beacon_model = result.scalar_one_or_none()
        if beacon_model and beacon_model.state == "undiscovered":
            await _transition_beacon(beacon_model.id, "discovered", db)
        if not beacon_model or not beacon_model.spirit_id:
            # No spirit associated, show generic page
            template = jinja_env.get_template("found_no_spirit.html")
            return HTMLResponse(content=template.render(beacon_id=id))

        # Get the spirit
        spirit_result = await db.execute(
            select(SpiritModel).where(SpiritModel.id == beacon_model.spirit_id)
        )
        spirit_model = spirit_result.scalar_one_or_none()
        if not spirit_model:
            template = jinja_env.get_template("found_no_spirit.html")
            return HTMLResponse(content=template.render(beacon_id=id))

        # Redirect to spirit page with from_beacon flag
        return RedirectResponse(
            f"/spirit/{spirit_model.id}?from_beacon=true", status_code=status.HTTP_303_SEE_OTHER
        )

    @app.get("/spirit/{spirit_id}", response_class=HTMLResponse)
    async def spirit_page(
        spirit_id: str,
        from_beacon: bool = False,
        db: AsyncSession = Depends(get_db_session),
    ):
        """Display the spirit info page.

        Args:
            spirit_id: The ID of the spirit to display
            from_beacon: If True, indicates the user scanned a beacon and should see the call to action
        """
        result = await db.execute(select(SpiritModel).where(SpiritModel.id == spirit_id))
        spirit_model = result.scalar_one_or_none()
        if not spirit_model:
            raise HTTPException(status_code=404, detail="Spirit not found")
        template = jinja_env.get_template("spirit.html")
        return HTMLResponse(
            content=template.render(
                spirit_name=spirit_model.name,
                blurb=spirit_model.blurb,
                image_url=spirit_model.image_url or "",
                totem_image_url=spirit_model.totem_image_url or "",
                current_activity=spirit_model.current_activity or "",
                from_beacon=from_beacon,
            )
        )

    @app.get("/api/state")
    async def get_state():
        return await state.snapshot()

    @app.get("/healthz")
    async def healthcheck():
        return {"status": "ok"}

    @app.websocket("/ws/mobile")
    async def mobile_channel(websocket: WebSocket):
        await websocket.accept()
        client_id = str(uuid.uuid4())
        await registry.register_mobile(client_id, websocket)
        try:
            await websocket.send_json({"type": "welcome", "clientId": client_id})
            while True:
                # Drain incoming messages to keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            await registry.unregister_mobile(client_id)
        except Exception:
            await registry.unregister_mobile(client_id)

    @app.websocket("/ws/display")
    async def display_channel(websocket: WebSocket):
        await websocket.accept()
        display_id = str(uuid.uuid4())
        await registry.register_display(display_id, websocket)
        try:
            await websocket.send_json({"type": "welcome", "displayId": display_id})
            # Immediately push current snapshot to the newly connected display.
            await websocket.send_json(await state.snapshot())
            while True:
                # Displays are read-only in this prototype; drain incoming messages to keep the connection alive.
                await websocket.receive_text()
        except WebSocketDisconnect:
            await registry.unregister_display(display_id)
        except Exception:
            await registry.unregister_display(display_id)

    return app


def main():
    import uvicorn

    host = os.getenv("MIDNIGHT_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MIDNIGHT_SERVER_PORT", os.getenv("PORT", "9000")))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
