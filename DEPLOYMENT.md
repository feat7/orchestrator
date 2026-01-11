# Deployment Guide - chatbackend.com

## Prerequisites

- VPS with Ubuntu 22.04+ (2GB RAM minimum)
- Domain `chatbackend.com` pointing to server IP (A record)

---

## Step 1: Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker compose version
```

---

## Step 2: Clone and Configure

```bash
# Clone
git clone https://github.com/feat7/orchestrator.git
cd orchestrator

# Configure environment
cp .env.prod.example .env
nano .env
```

**Set these values in .env:**
- `POSTGRES_PASSWORD` - strong password
- `OPENAI_API_KEY` - your API key

---

## Step 3: Deploy

```bash
# Build and start
docker compose -f docker-compose.prod.yml up -d --build

# Run migrations
docker compose -f docker-compose.prod.yml exec api alembic upgrade head

# Seed mock data (optional)
docker compose -f docker-compose.prod.yml exec api python -m scripts.seed_mock_data
```

---

## Step 4: Verify

- Visit `https://chatbackend.com` - UI should load
- Visit `https://chatbackend.com/api/v1/health` - should return healthy

---

## Common Commands

```bash
# View logs
docker compose -f docker-compose.prod.yml logs -f

# Restart
docker compose -f docker-compose.prod.yml restart

# Stop
docker compose -f docker-compose.prod.yml down

# Update (after git pull)
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
```
