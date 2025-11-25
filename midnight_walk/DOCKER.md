## Dockerized Midnight Walk Server

This repo now ships a Docker workflow that packages the Python server into a single image.

### 1. Build the wheel and Docker image

```bash
./midnight_walk/build_server_image.sh
```

The script runs `bazelisk build //midnight_walk:midnight_walk_server_wheel`, stages the wheel
alongside `midnight_walk/docker/Dockerfile`, and builds an image tagged `midnight-walk-server`.

### 2. Run the container

```bash
docker run --rm -p 9000:9000 midnight-walk-server
```

Environment variables you can override:

- `MIDNIGHT_SERVER_HOST` – bind address (default `0.0.0.0`)
- `MIDNIGHT_SERVER_PORT` – port inside the container (default `9000`)
- `FORCE_HTTPS` – set to `1` to force an HTTPS redirect based on `X-Forwarded-Proto`
- `MIDNIGHT_ADMIN_USER` / `MIDNIGHT_ADMIN_PASSWORD` – credentials for the `/admin` portal
  (defaults: `curator`/`midnight`)
- `MIDNIGHT_ADMIN_SESSION_SECRET` – secret key for signing admin sessions (default `dev-secret`)

### Deploying to Heroku

1. Ensure the Heroku CLI is installed and you’re logged in (`heroku login`).
2. Optionally set the target app name (defaults to `midnight-walk`):

   ```bash
   export HEROKU_APP=my-midnight-app
   ```

3. Run the deployment script:

   ```bash
   ./midnight_walk/deploy_heroku.sh
   ```

   This builds the Bazel wheel, builds the Docker image, pushes it to
   `registry.heroku.com/<HEROKU_APP>/web`, and calls `heroku container:release web`.

Heroku terminates HTTPS at its routers, so disable the embedded nginx inside the dyno and
bind FastAPI to the dyno port:

```bash
heroku config:set \
  MIDNIGHT_SERVER_HOST=0.0.0.0 \
  FORCE_HTTPS=1 \
  -a "$HEROKU_APP"
```

After adding your custom domain via `heroku domains:add`, enable Automatic Certificate Management with:

```bash
heroku certs:auto:enable -a "$HEROKU_APP"
```

Heroku will provision/renew a trusted TLS cert for the domain automatically once DNS is
pointing at the app.
