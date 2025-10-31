#!/bin/bash
# Docker networking demonstrations for ML infrastructure

set -e

echo "=== Docker Networking Demonstrations ==="
echo

# Demo 1: Bridge Network
echo "Demo 1: Creating Bridge Network"
docker network create --driver bridge \
  --subnet 172.25.0.0/16 \
  --gateway 172.25.0.1 \
  ml-bridge-demo

echo "✓ Bridge network created"
echo

# Demo 2: Custom Network with Containers
echo "Demo 2: Creating Containers on Custom Network"
docker run -d --name ml-api-1 --network ml-bridge-demo nginx:alpine
docker run -d --name ml-api-2 --network ml-bridge-demo nginx:alpine

echo "✓ Containers created on custom network"
echo

# Demo 3: Service Discovery
echo "Demo 3: Testing Service Discovery"
docker exec ml-api-1 ping -c 3 ml-api-2
echo "✓ Service discovery working (containers can ping by name)"
echo

# Demo 4: Network Isolation
echo "Demo 4: Creating Isolated Internal Network"
docker network create --driver bridge \
  --subnet 172.26.0.0/16 \
  --internal \
  ml-internal-demo

docker run -d --name ml-db --network ml-internal-demo postgres:alpine
echo "✓ Internal network created (no external access)"
echo

# Demo 5: Multiple Networks
echo "Demo 5: Connecting Container to Multiple Networks"
docker network create ml-frontend
docker network create ml-backend

docker run -d --name ml-gateway --network ml-frontend nginx:alpine
docker network connect ml-backend ml-gateway

echo "✓ Container connected to multiple networks"
echo

# Demo 6: Network Inspection
echo "Demo 6: Network Inspection"
docker network inspect ml-bridge-demo --format '{{json .IPAM.Config}}'
echo
echo "✓ Network details displayed"
echo

# Cleanup
echo "Cleaning up demo resources..."
docker rm -f ml-api-1 ml-api-2 ml-db ml-gateway 2>/dev/null || true
docker network rm ml-bridge-demo ml-internal-demo ml-frontend ml-backend 2>/dev/null || true

echo "✓ Cleanup complete"
echo
echo "=== Demonstrations Complete ==="
