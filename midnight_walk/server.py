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
import json
import logging
import math
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
    SpiritModel,
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
MAX_DISTANCE_FOR_HUE_METERS = 1200.0  # Distances beyond this clamp the hue scale.
CLIENT_STALE_SECONDS = 30.0

ADMIN_USERNAME = os.getenv("MIDNIGHT_ADMIN_USER", "curator")
ADMIN_PASSWORD = os.getenv("MIDNIGHT_ADMIN_PASSWORD", "midnight")
ADMIN_SESSION_SECRET = os.getenv("MIDNIGHT_ADMIN_SESSION_SECRET", "dev-secret")
ADMIN_REALM = "Midnight Walk Admin"

BEACON_STATES = ("undiscovered", "discovered", "returned", "vanquished")

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
class ClientLocation:
    latitude: float
    longitude: float
    last_seen: float


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
    current_activity: str | None = None


class SpiritImportRequest(BaseModel):
    spirits: List[SpiritImportRecord]


class MidnightGameState:
    def __init__(
        self, default_location: Tuple[float, float], db_session: AsyncSession | None = None
    ):
        self._lock = asyncio.Lock()
        self._display_lat, self._display_lon = default_location
        self._clients: Dict[str, ClientLocation] = {}
        self._beacons: Dict[str, Beacon] = {}
        self._db_session = db_session

    async def set_display_location(self, lat: float, lon: float) -> None:
        async with self._lock:
            self._display_lat = lat
            self._display_lon = lon

    async def get_display_location(self) -> Tuple[float, float]:
        async with self._lock:
            return self._display_lat, self._display_lon

    async def update_client(
        self,
        client_id: str,
        lat: float,
        lon: float,
    ) -> None:
        async with self._lock:
            self._clients[client_id] = ClientLocation(lat, lon, time.time())

    async def remove_client(self, client_id: str) -> None:
        async with self._lock:
            self._clients.pop(client_id, None)

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

    async def get_client_locations(self) -> Dict[str, ClientLocation]:
        async with self._lock:
            self._prune_locked()
            return self._clients.copy()

    async def snapshot(self) -> Dict[str, object]:
        async with self._lock:
            self._prune_locked()
            mean_distance = self._mean_distance_locked()
            color_rgb = hue_to_rgb(self._distance_to_hue(mean_distance))
            hex_color = "#{:02x}{:02x}{:02x}".format(*color_rgb)
            return {
                "serverTime": time.time(),
                "displayLocation": {
                    "lat": self._display_lat,
                    "lon": self._display_lon,
                },
                "activeClients": len(self._clients),
                "meanDistanceMeters": mean_distance,
                "color": {
                    "rgb": {"r": color_rgb[0], "g": color_rgb[1], "b": color_rgb[2]},
                    "hex": hex_color,
                },
            }

    def _prune_locked(self) -> None:
        now = time.time()
        stale = [
            cid for cid, loc in self._clients.items() if now - loc.last_seen > CLIENT_STALE_SECONDS
        ]
        for cid in stale:
            self._clients.pop(cid, None)

    def _distance_to_hue(self, distance_m: float) -> float:
        if distance_m <= 0:
            return 180.0  # cyan when people are clustered at the display.
        clamped = min(distance_m, MAX_DISTANCE_FOR_HUE_METERS)
        ratio = clamped / MAX_DISTANCE_FOR_HUE_METERS
        # Map 0m -> 180deg (cyan) and MAX_DISTANCE -> 0deg (red).
        return max(0.0, 180.0 * (1.0 - ratio))

    def _mean_distance_locked(self) -> float:
        if not self._clients:
            return 0.0
        accum = 0.0
        count = 0
        for loc in self._clients.values():
            accum += haversine_meters(
                self._display_lat,
                self._display_lon,
                loc.latitude,
                loc.longitude,
            )
            count += 1
        return accum / max(count, 1)


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


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon pairs in meters."""
    radius = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def hue_to_rgb(
    hue_degrees: float, saturation: float = 1.0, value: float = 1.0
) -> Tuple[int, int, int]:
    """Convert HSV to RGB (0-255)."""
    hue_degrees %= 360
    c = value * saturation
    x = c * (1 - abs((hue_degrees / 60.0) % 2 - 1))
    m = value - c
    if hue_degrees < 60:
        r, g, b = c, x, 0
    elif hue_degrees < 120:
        r, g, b = x, c, 0
    elif hue_degrees < 180:
        r, g, b = 0, c, x
    elif hue_degrees < 240:
        r, g, b = 0, x, c
    elif hue_degrees < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (
        int((r + m) * 255),
        int((g + m) * 255),
        int((b + m) * 255),
    )


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
        lat, lon = await state.get_display_location()
        return {"lat": lat, "lon": lon}

    @app.post("/api/display-location", dependencies=[Depends(admin_guard)])
    async def set_display_location(payload: DisplayLocationPayload):
        await state.set_display_location(payload.lat, payload.lon)
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

        updated = False
        if payload.spirit_id is not None:
            beacon_model.spirit_id = payload.spirit_id
            updated = True

        if payload.state is not None:
            if payload.state not in BEACON_STATES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid state '{payload.state}'",
                )
            changed = await state.set_beacon_state(beacon_id, payload.state, db_session=db)
            updated = updated or changed

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

        if updated:
            await push_state()
        return {"status": "ok", "updated": updated}

    async def _apply_state_transition(
        beacon_id: str,
        new_state: str,
        db: AsyncSession,
    ):
        updated = await state.set_beacon_state(beacon_id, new_state, db_session=db)
        if not updated:
            raise HTTPException(status_code=404, detail="Beacon not found")
        await push_state()
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
        return await _apply_state_transition(beacon_id, "returned", db)

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
        spirit_model.current_activity = payload.current_activity
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
            if await state.set_beacon_state(beacon_model.id, "discovered", db_session=db):
                await push_state()
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

        # Redirect to spirit page
        return RedirectResponse(f"/spirit/{spirit_model.id}", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/spirit/{spirit_id}", response_class=HTMLResponse)
    async def spirit_page(spirit_id: str, db: AsyncSession = Depends(get_db_session)):
        """Display the spirit info page."""
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
                current_activity=spirit_model.current_activity or "",
            )
        )

    @app.get("/api/client-locations", dependencies=[Depends(admin_guard)])
    async def get_client_locations():
        clients = await state.get_client_locations()
        return {
            "clients": [
                {
                    "id": client_id,
                    "lat": loc.latitude,
                    "lon": loc.longitude,
                    "lastSeen": loc.last_seen,
                }
                for client_id, loc in clients.items()
            ]
        }

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
                message = await websocket.receive_text()
                data = json.loads(message)
                if data.get("type") == "location":
                    await state.update_client(client_id, float(data["lat"]), float(data["lon"]))
                    await push_state()
        except WebSocketDisconnect:
            await registry.unregister_mobile(client_id)
            await state.remove_client(client_id)
            await push_state()
        except Exception:
            await registry.unregister_mobile(client_id)
            await state.remove_client(client_id)
            await push_state()

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
