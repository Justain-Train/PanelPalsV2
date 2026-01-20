#!/bin/bash
# Quick start script for PanelPals V2 backend

set -e

echo "🚀 PanelPals V2 - Quick Start"
echo "=============================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please edit it with your API keys:"
        echo "   - GOOGLE_CLOUD_PROJECT"
        echo "   - GOOGLE_APPLICATION_CREDENTIALS"
        echo "   - ELEVENLABS_API_KEY"
        echo ""
        read -p "Press Enter after configuring .env to continue..."
    else
        echo "❌ .env.example not found. Please create .env manually."
        exit 1
    fi
fi

echo "🔨 Building Docker images..."
docker-compose build

echo ""
echo "🚀 Starting containers..."
docker-compose up -d

echo ""
echo "⏳ Waiting for backend to be ready..."
sleep 5

# Check if backend is healthy
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ Backend is healthy and running!"
    echo ""
    echo "📍 Backend URL: http://localhost:8000"
    echo "📖 API Docs: http://localhost:8000/docs"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs:    docker-compose logs -f backend"
    echo "   Stop:         docker-compose down"
    echo "   Restart:      docker-compose restart backend"
    echo "   Shell:        docker-compose exec backend bash"
    echo "   Or use:       make help"
else
    echo "⚠️  Backend is starting but not ready yet."
    echo "   Check logs with: docker-compose logs -f backend"
fi
