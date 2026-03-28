# Running Sancta in Docker on Ubuntu VM

Run the Sancta SIEM dashboard and agent in Docker on your Ubuntu VM to avoid Windows-specific crashes (ACCESS_VIOLATION, psutil, etc.).

## Prerequisites

- Ubuntu VM with Docker and Docker Compose installed
- Project files on the VM (clone or copy from your Windows machine)

## Quick Start

### 1. Copy project to your Ubuntu VM

From your Windows machine, copy the project to the VM (e.g. via SCP, shared folder, or git clone):

```bash
# On VM: clone from GitHub (if pushed)
git clone https://github.com/Mosalah992/sancta.git
cd sancta
```

Or use SCP from Windows PowerShell:

```powershell
scp -r "E:\CODE PROKECTS\sancta-main\sancta-main" user@vm-ip:~/sancta
```

### 2. Create `.env` on the VM

Copy your `.env` from Windows or create one on the VM:

```bash
cd ~/sancta  # or wherever you put the project
cp .env.example .env
nano .env    # edit: AGENT_NAME, MOLTBOOK_API_KEY (or leave blank to register)
```

Minimum `.env`:

```
AGENT_NAME=my-cool-agent
MOLTBOOK_API_KEY=
MOLTBOOK_CLAIM_URL=
HEARTBEAT_INTERVAL_MINUTES=30
```

Leave `MOLTBOOK_API_KEY` blank on first run to register via the dashboard.

### 3. Build and run

```bash
docker compose up -d --build
```

First build can take 5–15 minutes (PyTorch, sentence-transformers, etc.).

### 4. Access the dashboard

Open in your browser:

```
http://<vm-ip>:8787
```

Replace `<vm-ip>` with your Ubuntu VM’s IP (e.g. `192.168.1.100`).

### 5. Start the agent

1. Open the SIEM dashboard
2. Click **START** to run the Sancta agent
3. If you need to register, use **--register** first (see below)

## Useful commands

```bash
# View logs
docker compose logs -f

# Stop
docker compose down

# Rebuild after code changes
docker compose up -d --build

# Run agent registration only (one-off)
docker compose run --rm sancta python -m backend.sancta --register
```

## First-time registration

If `MOLTBOOK_API_KEY` is empty:

1. Run: `docker compose run --rm sancta python -m backend.sancta --register`
2. Copy the claim URL from the output
3. Complete the claim flow (e.g. tweet verification)
4. The script writes the API key to `.env`
5. Start the dashboard and agent as usual

## Persistence

The project directory is bind-mounted (`.:/app`), so:

- `agent_state.json`, `knowledge_db.json`, `.agent.pid` persist on the host
- `logs/` and `data/` persist on the host
- No need for extra volumes

## Optional: lighter image (CPU-only PyTorch)

To reduce image size (~1.5GB smaller), edit the Dockerfile and uncomment:

```dockerfile
RUN pip uninstall -y torch && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
```

Then rebuild: `docker compose build --no-cache`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 8787 in use | Change `ports: "8787:8787"` to e.g. `"8888:8787"` in docker-compose.yml |
| Can't reach dashboard | Check VM firewall: `sudo ufw allow 8787` |
| Agent stops after embedding load | Normal on Linux; if it still stops, check `docker compose logs` |
| Out of memory | Use CPU-only PyTorch (see above) or increase VM RAM |
