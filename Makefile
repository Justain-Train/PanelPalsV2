.PHONY: help build up down logs restart shell test clean prod-build prod-up

help:
	@echo "PanelPals V2 - Docker Commands"
	@echo ""
	@echo "Development:"
	@echo "  make build       - Build Docker images"
	@echo "  make up          - Start containers"
	@echo "  make down        - Stop containers"
	@echo "  make logs        - View container logs"
	@echo "  make restart     - Restart containers"
	@echo "  make shell       - Open shell in backend container"
	@echo "  make test        - Run tests in container"
	@echo "  make clean       - Remove containers and volumes"
	@echo ""
	@echo "Production:"
	@echo "  make prod-build  - Build production image"
	@echo "  make prod-up     - Run production container"
	@echo ""

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Backend running at http://localhost:8000"
	@echo "View logs with: make logs"

down:
	docker-compose down

logs:
	docker-compose logs -f backend

restart:
	docker-compose restart backend

shell:
	docker-compose exec backend bash

test:
	docker-compose exec backend pytest tests/ -v

clean:
	docker-compose down -v
	docker system prune -f

prod-build:
	docker build -f Dockerfile.production -t panelpals-backend:latest .

prod-up:
	docker run -d \
		--name panelpals-backend-prod \
		-p 8000:8000 \
		--env-file .env \
		--restart unless-stopped \
		panelpals-backend:latest
	@echo "Production backend running at http://localhost:8000"
