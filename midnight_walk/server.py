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
from dataclasses import dataclass
from typing import Dict, Tuple

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
from pydantic import BaseModel, Field
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.sessions import SessionMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

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


@dataclass
class ClientLocation:
    latitude: float
    longitude: float
    last_seen: float


class DisplayLocationPayload(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


class MidnightGameState:
    def __init__(self, default_location: Tuple[float, float]):
        self._lock = asyncio.Lock()
        self._display_lat, self._display_lon = default_location
        self._clients: Dict[str, ClientLocation] = {}

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


MOBILE_PAGE = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Midnight Renegade Walk</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      :root {
        color-scheme: dark;
        font-family: "Space Mono", "SF Pro Display", system-ui, -apple-system, sans-serif;
        background-color: #050505;
        color: #f5f5f5;
      }
      body {
        margin: 0;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        padding: 1.5rem;
        gap: 1.5rem;
      }
      header {
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-size: 1.1rem;
      }
      section {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
      }
      #color-swatch {
        width: 100%;
        height: 180px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: linear-gradient(135deg, #222, #000);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
      }
      button {
        padding: 0.75rem 1.25rem;
        border-radius: 999px;
        border: none;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        background: #f5f5f5;
        color: #050505;
      }
      .status {
        font-size: 0.95rem;
        opacity: 0.85;
        line-height: 1.6;
      }
      .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        font-size: 0.85rem;
        letter-spacing: 0.08em;
      }
    </style>
  </head>
  <body>
    <header>Midnight Renegade · Art Walk</header>
    <section>
      <div class="pill" id="connection-pill">CONNECTING…</div>
      <p class="status" id="status-text">
        Stay close to the volumetric beacon. Your phone whispers your position to the
        renegade host to influence the light.
      </p>
      <div id="color-swatch">Awaiting signal</div>
      <p class="status" id="distance-readout">Mean distance: –</p>
      <button id="retry-btn" hidden>Retry Connection</button>
    </section>
    <script>
      const statusText = document.getElementById("status-text");
      const colorSwatch = document.getElementById("color-swatch");
      const distanceReadout = document.getElementById("distance-readout");
      const pill = document.getElementById("connection-pill");
      const retry = document.getElementById("retry-btn");

      let ws;
      let watchId;

      function connect() {
        const protocol = location.protocol === "https:" ? "wss" : "ws";
        ws = new WebSocket(protocol + "://" + location.host + "/ws/mobile");
        ws.onopen = () => {
          pill.textContent = "CONNECTED";
          pill.style.background = "rgba(0, 255, 200, 0.15)";
          retry.hidden = true;
          startGeolocation();
        };
        ws.onclose = () => {
          pill.textContent = "DISCONNECTED";
          pill.style.background = "rgba(255, 60, 60, 0.2)";
          retry.hidden = false;
          stopGeolocation();
        };
        ws.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data);
            if (payload.color) {
              colorSwatch.style.background = payload.color.hex;
              colorSwatch.textContent = payload.color.hex;
            }
            if (payload.meanDistanceMeters != null) {
              const meters = payload.meanDistanceMeters;
              distanceReadout.textContent = "Mean distance: " + meters.toFixed(1) + " m";
            }
            if (payload.activeClients != null) {
              statusText.textContent = `You're synced with ${payload.activeClients} walkers.`;
            }
          } catch (err) {
            console.error("Unable to parse message", err);
          }
        };
        ws.onerror = (err) => {
          console.error("WebSocket error", err);
          ws.close();
        };
      }

      function startGeolocation() {
        if (!navigator.geolocation) {
          statusText.textContent = "Geolocation unavailable.";
          return;
        }
        watchId = navigator.geolocation.watchPosition(
          (position) => {
            const payload = {
              type: "location",
              lat: position.coords.latitude,
              lon: position.coords.longitude,
              accuracy: position.coords.accuracy,
            };
            ws.send(JSON.stringify(payload));
          },
          (err) => {
            statusText.textContent = "Location error: " + err.message;
          },
          {
            enableHighAccuracy: true,
            maximumAge: 2000,
            timeout: 5000,
          }
        );
      }

      function stopGeolocation() {
        if (watchId != null) {
          navigator.geolocation.clearWatch(watchId);
          watchId = null;
        }
      }

      retry.addEventListener("click", () => {
        connect();
      });

      connect();
    </script>
  </body>
</html>
"""


ADMIN_PAGE = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Midnight Renegade Admin</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link
      rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-sA+4J1N7lik6J9G3hYmt8DxXgPGwHNpBMpZ8Ff9i9oA="
      crossorigin=""
    />
    <style>
      :root {
        font-family: "Space Mono", "Segoe UI", sans-serif;
        background: #050505;
        color: #fff;
      }
      body {
        margin: 0;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
      }
      header {
        padding: 1rem 1.5rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }
      #map {
        flex: 1;
      }
      .panel {
        padding: 1rem 1.5rem 2rem;
        background: rgba(0, 0, 0, 0.65);
        border-top: 1px solid rgba(255, 255, 255, 0.08);
      }
      .panel button {
        padding: 0.6rem 1.5rem;
        border-radius: 999px;
        border: none;
        font-size: 0.95rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }
      .stat-line {
        font-size: 0.95rem;
        margin: 0.15rem 0;
        opacity: 0.85;
      }
    </style>
  </head>
  <body>
    <header>Admin · Beacon Location</header>
    <div id="map"></div>
    <div class="panel">
      <p class="stat-line">Active walkers: <span id="walker-count">0</span></p>
      <p class="stat-line">Mean distance: <span id="mean-distance">0 m</span></p>
      <button id="center-btn">Center On Display</button>
    </div>

    <script
      src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
      integrity="sha256-p4QY9Rrq6ArtyTcwxgypKekJy5o+1OtMQS8gZ6b3G0k="
      crossorigin=""
    ></script>
    <script>
      const map = L.map("map");
      const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "© OpenStreetMap contributors",
      });
      tiles.addTo(map);

      let marker;

      async function fetchDisplayLocation() {
        const response = await fetch("/api/display-location");
        if (!response.ok) {
          throw new Error("Unable to fetch location");
        }
        return await response.json();
      }

      async function updateStats() {
        const response = await fetch("/api/state");
        if (!response.ok) return;
        const payload = await response.json();
        document.getElementById("walker-count").textContent = payload.activeClients;
        document.getElementById("mean-distance").textContent = payload.meanDistanceMeters.toFixed(1) + " m";
      }

      async function init() {
        const loc = await fetchDisplayLocation();
        const latLng = [loc.lat, loc.lon];
        map.setView(latLng, 16);
        marker = L.marker(latLng, { draggable: true }).addTo(map);
        marker.bindPopup("Volumetric Module");
        marker.on("dragend", async () => {
          const position = marker.getLatLng();
          await sendUpdate(position.lat, position.lng);
        });
        map.on("click", async (event) => {
          marker.setLatLng(event.latlng);
          await sendUpdate(event.latlng.lat, event.latlng.lng);
        });
        setInterval(updateStats, 4000);
      }

      async function sendUpdate(lat, lon) {
        await fetch("/api/display-location", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ lat, lon }),
        });
      }

      document.getElementById("center-btn").addEventListener("click", async () => {
        const loc = await fetchDisplayLocation();
        map.setView([loc.lat, loc.lon], 16);
        marker.setLatLng([loc.lat, loc.lon]);
      });

      init();
    </script>
  </body>
</html>
"""

ADMIN_LOGIN_PAGE = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Admin Login · Midnight Walk</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      :root {{
        font-family: "Space Mono", "Segoe UI", sans-serif;
        background: #050505;
        color: #fff;
      }}
      body {{
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0;
      }}
      form {{
        padding: 2rem;
        background: rgba(0, 0, 0, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        width: min(360px, 90vw);
        display: flex;
        flex-direction: column;
        gap: 1rem;
      }}
      label {{
        display: flex;
        flex-direction: column;
        font-size: 0.9rem;
        gap: 0.4rem;
      }}
      input {{
        padding: 0.6rem 0.8rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        background: rgba(0, 0, 0, 0.4);
        color: #fff;
      }}
      button {{
        padding: 0.75rem 1.25rem;
        border: none;
        border-radius: 999px;
        background: #00ffc6;
        color: #050505;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        cursor: pointer;
      }}
      .error {{
        color: #ff6b6b;
        font-size: 0.85rem;
        min-height: 1rem;
      }}
    </style>
  </head>
  <body>
    <form method="post" action="/admin/login">
      <h2>Admin Login</h2>
      <label>
        Username
        <input name="username" autocomplete="username" required />
      </label>
      <label>
        Password
        <input type="password" name="password" autocomplete="current-password" required />
      </label>
      <div class="error">{error_msg}</div>
      <button type="submit">Enter</button>
    </form>
  </body>
</html>
"""


def create_app() -> FastAPI:
    app = FastAPI(title="Midnight Renegade Server", version="0.1.0")
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

    state = MidnightGameState(DEFAULT_DISPLAY_LOCATION)
    registry = ConnectionRegistry()

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
        return HTMLResponse(content=MOBILE_PAGE)

    def is_admin(request: Request) -> bool:
        return request.session.get("admin_authenticated") is True

    @app.get("/admin/login", response_class=HTMLResponse)
    async def admin_login_page(request: Request):
        if is_admin(request):
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
        return HTMLResponse(content=ADMIN_LOGIN_PAGE.format(error_msg=""))

    @app.post("/admin/login")
    async def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            request.session["admin_authenticated"] = True
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
        return HTMLResponse(
            content=ADMIN_LOGIN_PAGE.format(error_msg="Invalid credentials."),
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
        return HTMLResponse(content=ADMIN_PAGE)

    @app.get("/api/display-location")
    async def get_display_location():
        lat, lon = await state.get_display_location()
        return {"lat": lat, "lon": lon}

    @app.post("/api/display-location", dependencies=[Depends(admin_guard)])
    async def set_display_location(payload: DisplayLocationPayload):
        await state.set_display_location(payload.lat, payload.lon)
        await push_state()
        return {"status": "ok"}

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
