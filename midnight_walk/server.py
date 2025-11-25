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
from typing import AsyncGenerator, Dict, Tuple

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
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.sessions import SessionMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from midnight_walk.database import (
    BeaconModel,
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


class DisplayLocationPayload(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


class BeaconPayload(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    search_radius_meters: float = Field(..., gt=0.0, le=10000.0)


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
    ) -> None:
        async with self._lock:
            self._beacons[beacon_id] = Beacon(beacon_id, lat, lon, search_radius_meters)
            # Persist to database
            session = db_session or self._db_session
            if session:
                beacon_model = BeaconModel(
                    id=beacon_id,
                    latitude=lat,
                    longitude=lon,
                    search_radius_meters=search_radius_meters,
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
    ) -> None:
        async with self._lock:
            if beacon_id in self._beacons:
                self._beacons[beacon_id] = Beacon(beacon_id, lat, lon, search_radius_meters)
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
                        await session.commit()

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
                        model.id, model.latitude, model.longitude, model.search_radius_meters
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
                    model.id, model.latitude, model.longitude, model.search_radius_meters
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
                }
                for b in beacons.values()
            ]
        }

    @app.post("/api/beacons", dependencies=[Depends(admin_guard)])
    async def create_beacon(payload: BeaconPayload, db: AsyncSession = Depends(get_db_session)):
        beacon_id = str(uuid.uuid4())
        await state.add_beacon(
            beacon_id, payload.lat, payload.lon, payload.search_radius_meters, db_session=db
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
