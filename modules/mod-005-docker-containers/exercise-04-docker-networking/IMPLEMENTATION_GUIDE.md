# Implementation Guide: Docker Networking for ML Infrastructure

**Module**: MOD-005 Docker Containers
**Exercise**: 04 - Docker Networking Scenarios
**Difficulty**: Intermediate to Advanced
**Estimated Time**: 3-4 hours

---

## Table of Contents

1. [Introduction](#introduction)
2. [Docker Network Types](#docker-network-types)
3. [Container-to-Container Communication](#container-to-container-communication)
4. [External Connectivity and Port Mapping](#external-connectivity-and-port-mapping)
5. [Network Isolation and Security](#network-isolation-and-security)
6. [DNS Resolution in Docker](#dns-resolution-in-docker)
7. [Load Balancing and Service Discovery](#load-balancing-and-service-discovery)
8. [Production ML Networking Patterns](#production-ml-networking-patterns)
9. [Network Debugging and Troubleshooting](#network-debugging-and-troubleshooting)
10. [Best Practices](#best-practices)

---

## Introduction

### What This Guide Covers

Docker networking is fundamental to building distributed ML systems. This guide teaches you how to:

- Configure different network drivers for various use cases
- Enable secure communication between containers
- Implement network isolation for security boundaries
- Design scalable networking architectures for ML workloads
- Debug and troubleshoot network issues

### Prerequisites

- Completed Exercises 01-03 (Docker fundamentals, images, and Compose)
- Basic understanding of networking concepts (IP addresses, ports, DNS)
- Docker installed and running
- Basic Linux command-line skills

### Learning Approach

Each section includes:
- **Concept Explanation**: Theory and use cases
- **Step-by-Step Implementation**: Practical examples
- **ML Context**: How it applies to ML infrastructure
- **Common Pitfalls**: What to watch out for
- **Verification Steps**: How to confirm it works

---

## Docker Network Types

### 1.1 Overview of Network Drivers

Docker provides several network drivers, each designed for specific use cases:

| Driver | Use Case | Isolation | DNS | Performance |
|--------|----------|-----------|-----|-------------|
| Bridge | Single-host containers | Medium | Yes (custom) | Good |
| Host | High-performance needs | None | N/A | Excellent |
| Overlay | Multi-host swarms | High | Yes | Good |
| Macvlan | Legacy app integration | High | No | Excellent |
| None | Maximum isolation | Complete | No | N/A |

### 1.2 Bridge Networks (Default for Containers)

Bridge networks create a software bridge between containers and the host, enabling container-to-container communication on a single host.

#### Default Bridge Network

**Implementation:**

```bash
# List existing networks
docker network ls

# Inspect default bridge network
docker network inspect bridge

# Run containers on default bridge
docker run -d --name web1 nginx:alpine
docker run -d --name web2 nginx:alpine

# Check IP addresses
docker inspect web1 --format='{{.NetworkSettings.IPAddress}}'
docker inspect web2 --format='{{.NetworkSettings.IPAddress}}'
```

**Key Limitation - DNS Resolution:**

```bash
# Try to ping by container name (will fail)
docker exec web1 ping -c 3 web2
# Output: ping: web2: Name or service not known

# Get IP of web2
WEB2_IP=$(docker inspect web2 --format='{{.NetworkSettings.IPAddress}}')

# Ping by IP works
docker exec web1 ping -c 3 $WEB2_IP
# Output: 3 packets transmitted, 3 received

# Cleanup
docker rm -f web1 web2
```

**Important**: The default bridge network does NOT support automatic DNS resolution between containers. You must use custom bridge networks for DNS functionality.

#### Custom Bridge Networks

Custom bridge networks provide automatic DNS resolution and better isolation.

**Implementation:**

```bash
# Create custom bridge network
docker network create ml-network

# Inspect the network
docker network inspect ml-network

# Note the subnet (typically 172.x.0.0/16)

# Run containers on custom network
docker run -d --name web1 --network ml-network nginx:alpine
docker run -d --name web2 --network ml-network nginx:alpine

# DNS resolution now works!
docker exec web1 ping -c 3 web2
# Output: 3 packets transmitted, 3 received

docker exec web2 ping -c 3 web1
# Success!
```

**ML Context**: In ML pipelines, use custom bridge networks to enable services to discover each other by name (e.g., `model-server`, `feature-store`, `training-coordinator`).

#### Custom Bridge with Specific Configuration

**Implementation:**

```bash
# Create network with custom subnet
docker network create \
  --driver bridge \
  --subnet 172.25.0.0/16 \
  --gateway 172.25.0.1 \
  --ip-range 172.25.5.0/24 \
  --opt "com.docker.network.bridge.name"="ml-bridge" \
  ml-custom-net

# Run containers with specific IPs
docker run -d \
  --name training-master \
  --network ml-custom-net \
  --ip 172.25.5.10 \
  nginx:alpine

docker run -d \
  --name training-worker-1 \
  --network ml-custom-net \
  --ip 172.25.5.11 \
  nginx:alpine

# Verify IPs
docker inspect training-master \
  --format='{{.NetworkSettings.Networks.ml_custom_net.IPAddress}}'
```

**When to Use Specific IPs**:
- Firewall rules require predictable addresses
- Integration with external monitoring systems
- Legacy applications expect specific IP ranges

### 1.3 Host Network

Host network mode removes network isolation, giving containers direct access to the host's network stack.

**Implementation:**

```bash
# Run container with host network
docker run -d --name web-host --network host nginx:alpine

# Container uses host's network directly
# No port mapping needed - nginx accessible on host's port 80

# Check network settings (note empty NetworkSettings)
docker inspect web-host --format='{{json .NetworkSettings.Networks}}'

# Test connectivity (if port 80 available)
curl http://localhost:80

# Cleanup
docker rm -f web-host
```

**Advantages**:
- No Network Address Translation (NAT) overhead
- Best performance for high-throughput applications
- Access to all host network interfaces
- Useful for network monitoring tools

**Disadvantages**:
- No network isolation (security risk)
- Can't run multiple containers on same port
- Less portable (host-specific configuration)
- Port conflicts with host services

**ML Context**: Use host networking for:
- High-performance inference servers requiring minimum latency
- Distributed training nodes needing maximum bandwidth
- Network monitoring containers (Prometheus node exporter)

### 1.4 None Network

Containers with no network access, providing maximum isolation.

**Implementation:**

```bash
# Create completely isolated container
docker run -d --name isolated --network none alpine sleep 3600

# Verify network interfaces (only loopback exists)
docker exec isolated ip addr show
# Output shows only 'lo' (127.0.0.1)

# No external connectivity
docker exec isolated ping -c 1 8.8.8.8
# Fails: Network is unreachable

# Cleanup
docker rm -f isolated
```

**Use Cases**:
- Batch processing jobs that don't need network access
- Security-sensitive workloads
- Testing scenarios
- Data processing containers that read/write to volumes only

**ML Context**: Use for:
- Offline batch inference reading from mounted volumes
- Data preprocessing containers
- Model training on local datasets without external data sources

### 1.5 Overlay Networks (Multi-Host)

Overlay networks enable container communication across multiple Docker hosts, essential for distributed systems.

**Conceptual Overview** (requires Docker Swarm):

```bash
# Initialize swarm (on manager node)
docker swarm init --advertise-addr <MANAGER-IP>

# Create overlay network
docker network create \
  --driver overlay \
  --attachable \
  distributed-ml-net

# Deploy services across the swarm
docker service create \
  --name training-cluster \
  --network distributed-ml-net \
  --replicas 3 \
  pytorch/pytorch:latest

# Containers on different hosts can communicate
# via the overlay network using DNS
```

**Features**:
- Automatic encryption (IPsec)
- Service discovery across hosts
- Load balancing built-in
- Supports thousands of containers

**ML Context**: Critical for:
- Multi-node distributed training
- Distributed inference clusters
- Federated learning systems
- Large-scale data processing pipelines

### 1.6 Macvlan Networks

Macvlan assigns a MAC address to containers, making them appear as physical devices on the network.

**Implementation:**

```bash
# Create macvlan network
# Note: Requires parent interface (e.g., eth0)
docker network create -d macvlan \
  --subnet=192.168.1.0/24 \
  --gateway=192.168.1.1 \
  -o parent=eth0 \
  macvlan-net

# Run container with macvlan
docker run -d \
  --name legacy-app \
  --network macvlan-net \
  --ip 192.168.1.100 \
  nginx:alpine
```

**Use Cases**:
- Legacy applications requiring Layer 2 access
- Integration with physical network devices
- VLAN-based network segmentation
- Applications expecting direct network access

**ML Context**: Rarely used in ML, but applicable for:
- Integration with legacy GPU management systems
- Specific hardware monitoring requirements
- Network-attached storage (NAS) systems

---

## Container-to-Container Communication

### 2.1 Communication Within Same Network

Containers on the same custom network communicate using DNS names.

**Implementation:**

```bash
# Create network
docker network create app-net

# Run a database
docker run -d \
  --name postgres-db \
  --network app-net \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=mldata \
  postgres:15-alpine

# Create a simple Python application
cat > app.py << 'EOF'
import psycopg2
import sys

try:
    # Connect using DNS name 'postgres-db'
    conn = psycopg2.connect(
        host='postgres-db',  # DNS resolves to container
        database='mldata',
        user='postgres',
        password='secret',
        connect_timeout=3
    )
    print(f"✓ Successfully connected to database at 'postgres-db'!")
    print(f"Connection info: {conn.get_dsn_parameters()}")

    # Test query
    cur = conn.cursor()
    cur.execute('SELECT version();')
    version = cur.fetchone()[0]
    print(f"PostgreSQL version: {version}")

    conn.close()
    sys.exit(0)
except Exception as e:
    print(f"✗ Connection failed: {e}")
    sys.exit(1)
EOF

# Create Dockerfile for client
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
RUN pip install psycopg2-binary
COPY app.py .
CMD ["python", "app.py"]
EOF

# Build and run client
docker build -t db-client .
docker run --rm --network app-net db-client
```

**Expected Output:**
```
✓ Successfully connected to database at 'postgres-db'!
Connection info: {'user': 'postgres', 'dbname': 'mldata', 'host': 'postgres-db', ...}
PostgreSQL version: PostgreSQL 15.x ...
```

### 2.2 Communication Across Multiple Networks

A container can connect to multiple networks simultaneously.

**Implementation:**

```bash
# Create two networks
docker network create frontend-net
docker network create backend-net

# Run database on backend only
docker run -d \
  --name database \
  --network backend-net \
  -e POSTGRES_PASSWORD=secret \
  postgres:15-alpine

# Run API on both networks
docker run -d \
  --name api-server \
  --network backend-net \
  nginx:alpine

# Connect API to frontend network as well
docker network connect frontend-net api-server

# Run web server on frontend only
docker run -d \
  --name web-frontend \
  --network frontend-net \
  nginx:alpine

# Verify connectivity
# Web can reach API (both on frontend-net)
docker exec web-frontend ping -c 2 api-server
# ✓ Success

# Web CANNOT reach database (different networks)
docker exec web-frontend ping -c 2 database
# ✗ Fails

# API can reach database (both on backend-net)
docker exec api-server ping -c 2 database
# ✓ Success
```

**ML Context**: This pattern enables:
- Separating public-facing inference APIs from backend databases
- Isolating training infrastructure from production serving
- Creating security boundaries between data tiers

### 2.3 Network Aliases for Round-Robin Load Balancing

Multiple containers can share the same network alias for basic load balancing.

**Implementation:**

```bash
# Create network
docker network create lb-net

# Run 3 inference servers with same alias 'model-server'
docker run -d --name model-v1 \
  --network lb-net \
  --network-alias model-server \
  nginx:alpine

docker run -d --name model-v2 \
  --network lb-net \
  --network-alias model-server \
  nginx:alpine

docker run -d --name model-v3 \
  --network lb-net \
  --network-alias model-server \
  nginx:alpine

# Create client container
docker run -it --rm --network lb-net alpine sh

# Inside container, DNS resolution returns multiple IPs
# nslookup model-server
# You'll see 3 different IP addresses

# Multiple HTTP requests will round-robin
# for i in {1..10}; do wget -qO- http://model-server/; done
```

**How It Works**:
- DNS returns all IP addresses for the alias
- Client's resolver typically round-robins through the list
- Provides basic load distribution
- Not true health-aware load balancing

**ML Context**: Quick way to distribute inference requests across multiple model replicas without additional infrastructure.

---

## External Connectivity and Port Mapping

### 3.1 Port Publishing Basics

Docker maps container ports to host ports for external access.

**Port Mapping Syntax:**

```bash
# Basic syntax: -p HOST_PORT:CONTAINER_PORT

# Publish container's port 80 to host's 8080
docker run -d -p 8080:80 --name web nginx:alpine
# Access via http://localhost:8080

# Publish to specific interface (localhost only)
docker run -d -p 127.0.0.1:8081:80 --name web-local nginx:alpine
# Only accessible locally, not from network

# Let Docker assign random port
docker run -d -p 80 --name web-random nginx:alpine
docker port web-random
# Output: 80/tcp -> 0.0.0.0:32768 (or similar)

# Multiple port mappings
docker run -d \
  -p 8080:80 \
  -p 8443:443 \
  --name web-multi \
  nginx:alpine

# UDP ports
docker run -d -p 8125:8125/udp --name statsd hopsoft/graphite-statsd
```

### 3.2 The Expose Directive

The `EXPOSE` directive in Dockerfile documents which ports a container listens on, but does NOT publish them.

**Dockerfile Example:**

```dockerfile
FROM python:3.11-slim

# Document that app listens on port 5000
EXPOSE 5000

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY app.py .

# Run application
CMD ["python", "app.py"]
```

**Using in docker-compose.yml:**

```yaml
version: '3.8'

services:
  # Exposed externally via host port mapping
  web:
    build: .
    ports:
      - "8080:80"  # host:container
    # Accessible from host at localhost:8080

  # Exposed only to other containers
  api:
    build: ./api
    expose:
      - "5000"  # Documented, not published to host
    # Other services can access api:5000
    # NOT accessible from host

  # Internal service (no exposure)
  database:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=secret
    # No ports/expose
    # Only accessible via service name from other containers
```

**Key Differences**:
- `ports`: Maps container port to host port (external access)
- `expose`: Documents ports but doesn't publish (inter-container only)
- No directive: Port not documented, but still usable within same network

### 3.3 Host Network Access from Containers

Accessing services running on the host from containers.

**Implementation:**

```bash
# Modern Docker (18.03+) provides special DNS name
docker run --rm \
  --add-host host.docker.internal:host-gateway \
  alpine ping -c 3 host.docker.internal

# Access host's service from container
docker run -it --rm \
  --add-host host.docker.internal:host-gateway \
  curlimages/curl:latest \
  curl http://host.docker.internal:8000
```

**Use Cases**:
- Accessing development servers running on host
- Connecting to host's database during development
- Integration testing with host services

**ML Context**:
- Accessing GPU drivers on host (with --gpus flag)
- Connecting to host-based data sources
- Development/debugging scenarios

### 3.4 External Network Access

Containers access external networks through the host's network interface.

**Default Behavior:**

```bash
# Containers can reach internet by default
docker run --rm alpine ping -c 3 google.com
# Success

# Check DNS resolution
docker run --rm alpine nslookup github.com
# Success
```

**Custom DNS Configuration:**

```bash
# Specify DNS servers
docker run --rm \
  --dns 8.8.8.8 \
  --dns 8.8.4.4 \
  alpine nslookup github.com

# Add DNS search domains
docker run --rm \
  --dns-search example.com \
  alpine cat /etc/resolv.conf
```

**Docker Compose DNS Configuration:**

```yaml
version: '3.8'

services:
  app:
    image: alpine
    dns:
      - 8.8.8.8
      - 8.8.4.4
    dns_search:
      - company.internal
      - dev.company.internal
```

---

## Network Isolation and Security

### 4.1 Multi-Tier Network Architecture

Implementing defense-in-depth with network segmentation.

**3-Tier Architecture Example:**

```yaml
# docker-compose-3tier.yml
version: '3.8'

services:
  # TIER 1: DMZ (Demilitarized Zone) - Public-facing
  load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    networks:
      - dmz
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf

  # TIER 2: Application Layer
  web-app:
    build: ./webapp
    networks:
      - dmz          # Accessible from load balancer
      - app-tier     # Can communicate with API
    depends_on:
      - api

  api:
    build: ./api
    networks:
      - app-tier     # Receives from web-app
      - data-tier    # Can access database
    environment:
      - DB_HOST=database
      - CACHE_HOST=redis

  # TIER 3: Data Layer
  database:
    image: postgres:15-alpine
    networks:
      - data-tier    # Isolated from public
    environment:
      - POSTGRES_PASSWORD=secret
    volumes:
      - db-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    networks:
      - data-tier

networks:
  dmz:
    driver: bridge
    # Internet-facing

  app-tier:
    driver: bridge
    internal: true  # No external internet access

  data-tier:
    driver: bridge
    internal: true  # Maximum isolation

volumes:
  db-data:
```

**Security Benefits**:

1. **Lateral Movement Prevention**: Breach in DMZ doesn't directly expose data layer
2. **Minimized Attack Surface**: Database not accessible from internet
3. **Controlled Access Points**: API is only gateway to data
4. **Internal Network Isolation**: App and data tiers can't reach internet (no data exfiltration)

**Verification:**

```bash
# Start the stack
docker compose -f docker-compose-3tier.yml up -d

# Test connectivity

# 1. Load balancer CAN reach web-app
docker compose exec load-balancer nc -zv web-app 80
# ✓ Success

# 2. Web-app CAN reach API
docker compose exec web-app nc -zv api 5000
# ✓ Success

# 3. Web-app CANNOT reach database (different network)
docker compose exec web-app nc -zv database 5432
# ✗ Fails (timeout)

# 4. API CAN reach database
docker compose exec api nc -zv database 5432
# ✓ Success

# 5. Database CANNOT reach internet (internal network)
docker compose exec database ping -c 1 8.8.8.8
# ✗ Fails (network unreachable)

# 6. API CANNOT reach internet (internal network)
docker compose exec api curl http://google.com
# ✗ Fails
```

### 4.2 Frontend-Backend Separation

**Implementation:**

```yaml
# docker-compose-separation.yml
version: '3.8'

services:
  # Public-facing reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    networks:
      - frontend
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf

  # Application tier (bridge between networks)
  api:
    build: ./api
    networks:
      - frontend  # Accessible from nginx
      - backend   # Can access database
    environment:
      - DB_PASSWORD=${DB_PASSWORD}

  # Database (backend only - isolated)
  database:
    image: postgres:15-alpine
    networks:
      - backend   # Not on frontend network
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No internet access
```

**Nginx Configuration (nginx.conf):**

```nginx
upstream api_backend {
    server api:5000;
}

server {
    listen 80;
    server_name _;

    # Only allow specific endpoints
    location /api/v1/ {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
    }

    # Block direct database access attempts
    location /database {
        deny all;
        return 403;
    }
}
```

### 4.3 Internal Networks

Internal networks have no gateway to external networks, providing maximum isolation.

**Implementation:**

```bash
# Create internal network
docker network create \
  --internal \
  --subnet 172.30.0.0/16 \
  secure-internal

# Run services on internal network
docker run -d \
  --name secret-processor \
  --network secure-internal \
  alpine sleep 3600

# Verify no internet access
docker exec secret-processor ping -c 1 8.8.8.8
# Fails: network is unreachable

# Containers on same internal network can communicate
docker run -d --name processor-2 --network secure-internal alpine sleep 3600
docker exec processor-2 ping -c 2 secret-processor
# Success
```

**ML Context**: Use internal networks for:
- Sensitive training data processing
- Model parameter servers
- Internal feature stores
- Compliance requirements (data cannot leave environment)

### 4.4 Network Policies with Docker Compose

**Complete Isolated ML Pipeline:**

```yaml
version: '3.8'

services:
  # Public API endpoint
  inference-api:
    build: ./inference
    ports:
      - "8080:8080"
    networks:
      - public
      - inference-tier
    environment:
      - MODEL_SERVER=model-server:5000

  # Model serving (internal only)
  model-server:
    build: ./model-server
    networks:
      - inference-tier
      - ml-tier
    environment:
      - FEATURE_STORE=feature-store:6379

  # Feature store
  feature-store:
    image: redis:7-alpine
    networks:
      - ml-tier
      - data-tier

  # Training pipeline (completely isolated)
  training-coordinator:
    build: ./training
    networks:
      - ml-tier
      - data-tier
    volumes:
      - model-weights:/models

  # Data storage (deepest tier)
  postgres:
    image: postgres:15-alpine
    networks:
      - data-tier
    environment:
      - POSTGRES_PASSWORD=secret
    volumes:
      - training-data:/var/lib/postgresql/data

networks:
  public:
    driver: bridge
    # External access allowed

  inference-tier:
    driver: bridge
    internal: true

  ml-tier:
    driver: bridge
    internal: true

  data-tier:
    driver: bridge
    internal: true

volumes:
  model-weights:
  training-data:
```

**Network Diagram:**

```
Internet → [public] → inference-api
                         ↓
              [inference-tier] → model-server
                                    ↓
                         [ml-tier] → feature-store
                                    ↓               ↓
                                 training-coordinator
                                    ↓
                        [data-tier] → postgres
```

---

## DNS Resolution in Docker

### 5.1 How Docker DNS Works

Docker's embedded DNS server provides service discovery for containers.

**DNS Server Location**: `127.0.0.11` (internal DNS)

**Implementation:**

```bash
# Create custom network
docker network create test-dns

# Run container
docker run -d --name web --network test-dns nginx:alpine

# Check DNS configuration
docker exec web cat /etc/resolv.conf
```

**Output:**
```
nameserver 127.0.0.11
options ndots:0
```

**Key Points**:
- All containers on custom networks use 127.0.0.11 as DNS server
- Docker's DNS server resolves container names to IPs
- External DNS queries are forwarded to host's DNS servers

### 5.2 Service Discovery by Container Name

**Implementation:**

```bash
docker network create discovery-net

# Run services
docker run -d --name database --network discovery-net postgres:15-alpine \
  -e POSTGRES_PASSWORD=secret

docker run -d --name cache --network discovery-net redis:7-alpine

docker run -d --name api --network discovery-net nginx:alpine

# Test DNS resolution from any container
docker exec api nslookup database
# Returns IP of database container

docker exec api nslookup cache
# Returns IP of cache container

# Ping by name
docker exec api ping -c 2 database
# Success
```

### 5.3 Network Aliases

Containers can have multiple DNS names.

**Implementation:**

```bash
docker network create alias-net

# Container with multiple aliases
docker run -d --name web \
  --network alias-net \
  --network-alias webapp \
  --network-alias api \
  --network-alias frontend \
  nginx:alpine

# All aliases resolve to same container
docker run --rm --network alias-net alpine nslookup webapp
docker run --rm --network alias-net alpine nslookup api
docker run --rm --network alias-net alpine nslookup frontend
# All return same IP
```

**Use Cases**:
- Backward compatibility (old and new names)
- Multiple service roles (api + web)
- Migration scenarios

### 5.4 Docker Compose DNS

Compose provides automatic service discovery.

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  web:
    build: ./web
    environment:
      # Reference other services by their service name
      - API_URL=http://api:5000
      - DB_HOST=database
      - CACHE_HOST=redis
    depends_on:
      - api
      - database
      - redis

  api:
    build: ./api
    # Automatically gets DNS name 'api'
    environment:
      - DATABASE_URL=postgresql://postgres:secret@database:5432/app
      - REDIS_URL=redis://redis:6379

  database:
    image: postgres:15-alpine
    # Automatically gets DNS name 'database'
    environment:
      - POSTGRES_PASSWORD=secret

  redis:
    image: redis:7-alpine
    # Automatically gets DNS name 'redis'
```

**DNS Resolution in Compose**:
- Each service gets a DNS name matching its service name
- Services can reference each other reliably
- Names work across container restarts (IP changes don't matter)
- Compose creates a default network automatically

### 5.5 External DNS Configuration

**Global Docker Daemon Configuration:**

```bash
# Edit /etc/docker/daemon.json
sudo cat > /etc/docker/daemon.json << 'EOF'
{
  "dns": ["8.8.8.8", "8.8.4.4"],
  "dns-search": ["company.internal"],
  "dns-opts": ["ndots:2"]
}
EOF

# Restart Docker
sudo systemctl restart docker
```

**Per-Container DNS:**

```bash
docker run --rm \
  --dns 1.1.1.1 \
  --dns 8.8.8.8 \
  --dns-search example.com \
  --dns-opt ndots:1 \
  alpine cat /etc/resolv.conf
```

**Compose DNS Configuration:**

```yaml
version: '3.8'

services:
  app:
    image: alpine
    dns:
      - 8.8.8.8
      - 1.1.1.1
    dns_search:
      - company.internal
      - dev.company.internal
    dns_opt:
      - ndots:2
```

---

## Load Balancing and Service Discovery

### 6.1 DNS Round-Robin Load Balancing

**Simple Implementation:**

```bash
docker network create lb-net

# Run multiple backend services with same alias
for i in {1..3}; do
  docker run -d \
    --name backend-$i \
    --network lb-net \
    --network-alias backend \
    nginx:alpine
done

# Client sees round-robin DNS
docker run -it --rm --network lb-net alpine sh
# Inside: nslookup backend
# Shows 3 IP addresses

# Requests distribute across backends
for i in {1..10}; do
  docker run --rm --network lb-net curlimages/curl:latest \
    curl -s http://backend/
done
```

**Limitations**:
- No health checking
- Client-side load balancing (depends on client implementation)
- Sticky connections can cause imbalance
- Can't adjust weights

### 6.2 Nginx Load Balancer

**Directory Structure:**

```bash
mkdir -p lb-example/{nginx,app}
cd lb-example
```

**Backend Application (app/app.py):**

```python
from flask import Flask, jsonify
import socket
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'server': socket.gethostname(),
        'ip': socket.gethostbyname(socket.gethostname()),
        'version': os.environ.get('VERSION', '1.0')
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**app/requirements.txt:**

```
flask==3.0.0
```

**app/Dockerfile:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["python", "app.py"]
```

**Nginx Configuration (nginx/nginx.conf):**

```nginx
upstream backend_servers {
    # Load balancing method
    least_conn;  # Options: round_robin (default), least_conn, ip_hash

    # Backend servers with health check parameters
    server backend1:5000 max_fails=3 fail_timeout=30s;
    server backend2:5000 max_fails=3 fail_timeout=30s;
    server backend3:5000 max_fails=3 fail_timeout=30s;

    # Optional: weighted distribution
    # server backend1:5000 weight=3;
    # server backend2:5000 weight=2;
    # server backend3:5000 weight=1;
}

server {
    listen 80;
    server_name _;

    # Access logs
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    location / {
        proxy_pass http://backend_servers;

        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;

        # Retry on failure
        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_next_upstream_tries 2;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://backend_servers/health;
    }

    # Nginx stats
    location /nginx_status {
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        deny all;
    }
}
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - backend1
      - backend2
      - backend3
    networks:
      - lb-network

  backend1:
    build: ./app
    environment:
      - VERSION=1.0
    networks:
      - lb-network

  backend2:
    build: ./app
    environment:
      - VERSION=1.0
    networks:
      - lb-network

  backend3:
    build: ./app
    environment:
      - VERSION=1.0
    networks:
      - lb-network

networks:
  lb-network:
    driver: bridge
```

**Testing:**

```bash
# Start stack
docker compose up -d

# Test load balancing - observe different server hostnames
for i in {1..10}; do
  curl -s http://localhost/ | jq '.server'
  sleep 0.3
done

# Simulate backend failure
docker compose stop backend2

# Requests still work (distributed to healthy backends)
for i in {1..5}; do
  curl -s http://localhost/ | jq '.server'
done

# Check nginx status
docker compose exec nginx curl -s http://localhost/nginx_status

# Bring backend2 back online
docker compose start backend2

# Clean up
docker compose down
```

### 6.3 HAProxy Load Balancer

HAProxy provides advanced load balancing with health checks and statistics.

**haproxy.cfg:**

```cfg
global
    log stdout format raw local0
    maxconn 4096
    stats socket /var/run/haproxy.sock mode 660 level admin
    stats timeout 30s

defaults
    log     global
    mode    http
    option  httplog
    option  dontlognull
    option  http-server-close
    option  forwardfor except 127.0.0.0/8
    option  redispatch
    retries 3
    timeout connect 5s
    timeout client  50s
    timeout server  50s
    timeout queue   50s

# Frontend - entry point
frontend http_front
    bind *:80
    default_backend http_back

    # ACLs for routing
    acl is_api path_beg /api
    acl is_health path /health

    # Routing rules
    use_backend api_servers if is_api
    use_backend http_back if !is_api

# Backend - application servers
backend http_back
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200

    # Servers with health checks
    server backend1 backend1:5000 check inter 2s fall 3 rise 2
    server backend2 backend2:5000 check inter 2s fall 3 rise 2
    server backend3 backend3:5000 check inter 2s fall 3 rise 2

# Stats interface
listen stats
    bind *:8080
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-legends
    stats show-node
```

**docker-compose-haproxy.yml:**

```yaml
version: '3.8'

services:
  haproxy:
    image: haproxy:2.8-alpine
    ports:
      - "80:80"
      - "8080:8080"  # Stats page
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
    depends_on:
      - backend1
      - backend2
      - backend3
    networks:
      - lb-network

  backend1:
    build: ./app
    networks:
      - lb-network

  backend2:
    build: ./app
    networks:
      - lb-network

  backend3:
    build: ./app
    networks:
      - lb-network

networks:
  lb-network:
```

**Testing:**

```bash
# Start HAProxy stack
docker compose -f docker-compose-haproxy.yml up -d

# Test load balancing
for i in {1..10}; do
  curl -s http://localhost/ | jq '.server'
done

# View statistics dashboard
# Open browser to http://localhost:8080/stats

# Simulate failure
docker compose stop backend2

# HAProxy automatically removes unhealthy backend
# Requests continue working

# Check stats to see backend2 marked as DOWN
```

**Load Balancing Algorithms**:

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| roundrobin | Distribute sequentially | Equal capacity servers |
| leastconn | Send to server with fewest connections | Long-lived connections |
| source | Hash based on client IP | Session persistence |
| uri | Hash based on request URI | Cache efficiency |
| random | Random selection | Stateless services |

### 6.4 Service Discovery Patterns

**Consul-Based Discovery (Conceptual):**

```yaml
version: '3.8'

services:
  consul:
    image: consul:latest
    ports:
      - "8500:8500"
    networks:
      - discovery-net
    command: agent -server -ui -bootstrap-expect=1 -client=0.0.0.0

  registrator:
    image: gliderlabs/registrator:latest
    volumes:
      - /var/run/docker.sock:/tmp/docker.sock
    networks:
      - discovery-net
    command: -internal consul://consul:8500
    depends_on:
      - consul

  app:
    build: ./app
    networks:
      - discovery-net
    environment:
      - SERVICE_NAME=myapp
      - SERVICE_TAGS=production

networks:
  discovery-net:
```

**How It Works**:
1. Registrator watches Docker events
2. When container starts, registers it with Consul
3. Services query Consul to discover other services
4. Dynamic service updates

---

## Production ML Networking Patterns

### 7.1 Distributed Training Network Architecture

**Multi-Node Training Setup:**

```yaml
# docker-compose-distributed-training.yml
version: '3.8'

services:
  # Parameter server / master node
  master:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
    command: python train.py --role master --rank 0 --world-size 4
    volumes:
      - ./training:/workspace
      - model-checkpoints:/checkpoints
      - training-data:/data:ro
    networks:
      training-network:
        aliases:
          - master-node
    environment:
      - MASTER_ADDR=master-node
      - MASTER_PORT=29500
      - WORLD_SIZE=4
      - NCCL_SOCKET_IFNAME=eth0
      - NCCL_DEBUG=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Worker nodes
  worker-1:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
    command: python train.py --role worker --rank 1 --world-size 4
    volumes:
      - ./training:/workspace
      - model-checkpoints:/checkpoints
      - training-data:/data:ro
    networks:
      - training-network
    environment:
      - MASTER_ADDR=master-node
      - MASTER_PORT=29500
      - WORLD_SIZE=4
      - NCCL_SOCKET_IFNAME=eth0
    depends_on:
      - master
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  worker-2:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
    command: python train.py --role worker --rank 2 --world-size 4
    volumes:
      - ./training:/workspace
      - model-checkpoints:/checkpoints
      - training-data:/data:ro
    networks:
      - training-network
    environment:
      - MASTER_ADDR=master-node
      - MASTER_PORT=29500
      - WORLD_SIZE=4
      - NCCL_SOCKET_IFNAME=eth0
    depends_on:
      - master
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  worker-3:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
    command: python train.py --role worker --rank 3 --world-size 4
    volumes:
      - ./training:/workspace
      - model-checkpoints:/checkpoints
      - training-data:/data:ro
    networks:
      - training-network
    environment:
      - MASTER_ADDR=master-node
      - MASTER_PORT=29500
      - WORLD_SIZE=4
      - NCCL_SOCKET_IFNAME=eth0
    depends_on:
      - master
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Monitoring
  tensorboard:
    image: tensorflow/tensorflow:latest
    command: tensorboard --logdir=/logs --host=0.0.0.0 --port=6006
    ports:
      - "6006:6006"
    volumes:
      - tensorboard-logs:/logs
    networks:
      - training-network

networks:
  training-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
          gateway: 172.28.0.1

volumes:
  model-checkpoints:
  training-data:
  tensorboard-logs:
```

**Key Networking Considerations**:

1. **Consistent Network Interface**: `NCCL_SOCKET_IFNAME=eth0` ensures all nodes use same interface
2. **Master Address**: All workers connect to master via DNS name
3. **Port Consistency**: Port 29500 for communication
4. **Network Performance**: Custom subnet for predictable routing

### 7.2 Model Serving with A/B Testing

**Advanced Traffic Splitting:**

```yaml
# docker-compose-ab-testing.yml
version: '3.8'

services:
  # Nginx with A/B routing
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx-ab.conf:/etc/nginx/conf.d/default.conf:ro
    networks:
      - serving-network
    depends_on:
      - model-v1
      - model-v2

  # Model Version 1 (Stable - 80% traffic)
  model-v1:
    build:
      context: ./model-server
      args:
        MODEL_PATH: /models/v1
    environment:
      - MODEL_VERSION=v1
      - BATCH_SIZE=32
    volumes:
      - model-v1-weights:/models
    networks:
      serving-network:
        aliases:
          - model-a
    deploy:
      replicas: 4

  # Model Version 2 (Canary - 20% traffic)
  model-v2:
    build:
      context: ./model-server
      args:
        MODEL_PATH: /models/v2
    environment:
      - MODEL_VERSION=v2
      - BATCH_SIZE=32
    volumes:
      - model-v2-weights:/models
    networks:
      serving-network:
        aliases:
          - model-b
    deploy:
      replicas: 1

  # Metrics collection
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - serving-network
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Metrics visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - serving-network
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

networks:
  serving-network:
    driver: bridge

volumes:
  model-v1-weights:
  model-v2-weights:
  prometheus-data:
  grafana-data:
```

**Nginx A/B Configuration (nginx-ab.conf):**

```nginx
# Split traffic based on client IP
split_clients "${remote_addr}" $backend_pool {
    80%     model-a;
    20%     model-b;
}

upstream model-a {
    least_conn;
    server model-v1:8000 max_fails=3 fail_timeout=30s;
}

upstream model-b {
    least_conn;
    server model-v2:8000 max_fails=3 fail_timeout=30s;
}

# Logging for A/B analysis
log_format ab_testing '$remote_addr - $remote_user [$time_local] '
                      '"$request" $status $body_bytes_sent '
                      '"$http_referer" "$http_user_agent" '
                      'backend=$backend_pool';

server {
    listen 80;
    server_name _;

    access_log /var/log/nginx/ab_access.log ab_testing;

    location /predict {
        proxy_pass http://$backend_pool;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Backend-Pool $backend_pool;

        # Add model version to response headers
        add_header X-Model-Version $backend_pool always;
    }

    location /health {
        access_log off;
        return 200 "OK\n";
    }
}
```

### 7.3 Feature Store Architecture

**Online + Offline Feature Serving:**

```yaml
# docker-compose-feature-store.yml
version: '3.8'

services:
  # Online Feature API (low-latency serving)
  feature-api:
    build: ./feature-api
    ports:
      - "8080:8080"
    networks:
      - api-network
      - cache-network
    environment:
      - REDIS_HOST=redis-primary
      - POSTGRES_HOST=features-db
      - ENVIRONMENT=production
    depends_on:
      - redis-primary
      - features-db

  # Redis for fast feature access
  redis-primary:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    networks:
      - cache-network
      - storage-network
    volumes:
      - redis-data:/data

  # Redis replica for read scaling
  redis-replica:
    image: redis:7-alpine
    command: redis-server --replicaof redis-primary 6379
    networks:
      - cache-network
    depends_on:
      - redis-primary

  # PostgreSQL for feature metadata
  features-db:
    image: postgres:15-alpine
    networks:
      - storage-network
    environment:
      - POSTGRES_DB=features
      - POSTGRES_PASSWORD=secret
    volumes:
      - features-db-data:/var/lib/postgresql/data

  # Offline Feature Computation (Spark)
  spark-master:
    image: bitnami/spark:3.5
    networks:
      - compute-network
      - storage-network
    environment:
      - SPARK_MODE=master
      - SPARK_RPC_AUTHENTICATION_ENABLED=no
      - SPARK_RPC_ENCRYPTION_ENABLED=no
    ports:
      - "8081:8080"  # Spark UI
    volumes:
      - spark-events:/opt/spark/spark-events

  spark-worker-1:
    image: bitnami/spark:3.5
    networks:
      - compute-network
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_MEMORY=4G
      - SPARK_WORKER_CORES=2
    depends_on:
      - spark-master

  spark-worker-2:
    image: bitnami/spark:3.5
    networks:
      - compute-network
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_MEMORY=4G
      - SPARK_WORKER_CORES=2
    depends_on:
      - spark-master

  # Feature computation scheduler
  airflow:
    image: apache/airflow:2.7.0
    networks:
      - compute-network
      - storage-network
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://postgres:secret@features-db/features
    ports:
      - "8082:8080"
    volumes:
      - ./dags:/opt/airflow/dags
    depends_on:
      - features-db

networks:
  api-network:
    driver: bridge
    # External API access

  cache-network:
    driver: bridge
    internal: true
    # Fast feature cache layer

  storage-network:
    driver: bridge
    internal: true
    # Database access

  compute-network:
    driver: bridge
    internal: true
    # Batch feature computation

volumes:
  redis-data:
  features-db-data:
  spark-events:
```

**Network Segmentation Benefits**:

1. **api-network**: Public-facing, allows external requests
2. **cache-network**: Low-latency feature access, internal only
3. **storage-network**: Persistent storage tier, no direct public access
4. **compute-network**: Batch processing, completely isolated

### 7.4 Complete ML Stack with Monitoring

**Production-Ready ML Infrastructure:**

```yaml
# docker-compose-ml-stack.yml
version: '3.8'

services:
  # === INFERENCE TIER ===
  inference-api:
    build: ./inference-api
    ports:
      - "8000:8000"
    networks:
      - public-network
      - inference-network
    environment:
      - MODEL_SERVER_URL=http://model-server:5000
      - PROMETHEUS_MULTIPROC_DIR=/tmp
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G

  model-server:
    build: ./model-server
    networks:
      - inference-network
      - ml-network
    environment:
      - FEATURE_STORE_URL=http://feature-api:8080
    volumes:
      - model-weights:/models:ro
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # === MONITORING TIER ===
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    networks:
      - monitoring-network
      - inference-network
      - ml-network
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    networks:
      - monitoring-network
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    networks:
      - monitoring-network
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager-data:/alertmanager

  # === LOGGING TIER ===
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    networks:
      - monitoring-network
    volumes:
      - loki-data:/loki
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    networks:
      - monitoring-network
    volumes:
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - ./promtail.yml:/etc/promtail/config.yml:ro

  # === DATA TIER ===
  postgres:
    image: postgres:15-alpine
    networks:
      - data-network
    environment:
      - POSTGRES_PASSWORD=secret
      - POSTGRES_DB=mldata
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    networks:
      - data-network
    volumes:
      - redis-data:/data

networks:
  public-network:
    driver: bridge
    # Internet-facing

  inference-network:
    driver: bridge
    internal: true
    # Inference services

  ml-network:
    driver: bridge
    internal: true
    # ML operations

  monitoring-network:
    driver: bridge
    # Monitoring stack (needs external access for alerts)

  data-network:
    driver: bridge
    internal: true
    # Data storage

volumes:
  model-weights:
  prometheus-data:
  grafana-data:
  alertmanager-data:
  loki-data:
  postgres-data:
  redis-data:
```

---

## Network Debugging and Troubleshooting

### 8.1 Network Inspection Tools

**Basic Inspection:**

```bash
# List all networks
docker network ls

# Detailed network information
docker network inspect <network-name>

# Find containers on a network
docker network inspect <network-name> \
  --format='{{range .Containers}}{{.Name}} {{.IPv4Address}}{{"\n"}}{{end}}'

# Container's network configuration
docker inspect <container-name> \
  --format='{{json .NetworkSettings.Networks}}' | jq .

# Check container's IP address
docker inspect <container-name> \
  --format='{{.NetworkSettings.IPAddress}}'

# All network-related settings
docker inspect <container-name> --format='{{json .NetworkSettings}}' | jq .
```

### 8.2 Connectivity Testing

**Create Test Environment:**

```bash
# Create test network
docker network create debug-net

# Run server
docker run -d --name server --network debug-net nginx:alpine

# Run client with debugging tools
docker run -it --name client --network debug-net nicolaka/netshoot
```

**Inside netshoot container:**

```bash
# DNS resolution
nslookup server
dig server
host server

# Ping test
ping -c 3 server

# TCP connectivity
nc -zv server 80          # Port scan
telnet server 80          # Interactive

# HTTP testing
curl -v http://server/
wget -O- http://server/

# Network trace
traceroute server

# Check routing table
ip route show

# Network interfaces
ip addr show

# TCP/UDP connections
netstat -tunlp
ss -tunlp

# Packet capture
tcpdump -i eth0 -n
tcpdump -i eth0 -n port 80

# DNS debugging
nslookup server 127.0.0.11  # Query Docker's DNS directly
```

### 8.3 Common Issues and Solutions

**Issue 1: Containers Can't Communicate**

**Symptoms**: Connection timeouts, "no route to host"

**Diagnosis:**

```bash
# Check if containers are on same network
docker inspect container1 --format='{{json .NetworkSettings.Networks}}' | jq .
docker inspect container2 --format='{{json .NetworkSettings.Networks}}' | jq .

# Verify network exists
docker network ls | grep my-network

# Check network connectivity
docker exec container1 ping -c 2 container2
```

**Solutions:**

```bash
# Connect containers to same network
docker network connect my-network container1
docker network connect my-network container2

# Or recreate with correct network
docker run -d --name container1 --network my-network nginx
docker run -d --name container2 --network my-network nginx
```

**Issue 2: DNS Resolution Fails**

**Symptoms**: "Name or service not known", nslookup failures

**Diagnosis:**

```bash
# Check if using default bridge (DNS doesn't work there)
docker inspect container1 --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}'
# If output is "bridge" - that's the problem!

# Check DNS configuration
docker exec container1 cat /etc/resolv.conf

# Test DNS directly
docker exec container1 nslookup container2 127.0.0.11
```

**Solutions:**

```bash
# Use custom bridge network instead of default
docker network create my-custom-network
docker run -d --name container1 --network my-custom-network nginx
docker run -d --name container2 --network my-custom-network nginx

# Now DNS works
docker exec container1 ping container2
```

**Issue 3: Port Already in Use**

**Symptoms**: "Bind for 0.0.0.0:8080 failed: port is already allocated"

**Diagnosis:**

```bash
# Find what's using the port
sudo netstat -tlnp | grep :8080
# Or
sudo lsof -i :8080

# Check Docker containers
docker ps -a | grep 8080
docker port <container-name>
```

**Solutions:**

```bash
# Use different host port
docker run -d -p 8081:80 nginx

# Stop conflicting container
docker stop <conflicting-container>

# Kill process using port (if not Docker)
sudo kill <PID>
```

**Issue 4: Container Can't Reach Internet**

**Symptoms**: Can't ping external IPs, DNS fails for external domains

**Diagnosis:**

```bash
# Check if network is internal
docker network inspect my-network | grep internal

# Test DNS
docker exec container ping -c 1 8.8.8.8
docker exec container nslookup google.com

# Check Docker DNS settings
docker exec container cat /etc/resolv.conf
```

**Solutions:**

```bash
# If network is internal, recreate without that flag
docker network rm my-network
docker network create my-network  # Don't use --internal

# Configure DNS servers
docker run -d \
  --dns 8.8.8.8 \
  --dns 8.8.4.4 \
  nginx

# Or set in /etc/docker/daemon.json globally
{
  "dns": ["8.8.8.8", "8.8.4.4"]
}
sudo systemctl restart docker
```

**Issue 5: Network Performance Issues**

**Symptoms**: High latency, packet loss, slow transfers

**Diagnosis:**

```bash
# Network stats
docker stats

# Detailed network metrics
docker exec container1 iperf3 -s  # Server
docker exec container2 iperf3 -c container1  # Client

# Packet loss test
docker exec container1 ping -c 100 container2 | grep loss

# MTU issues
docker exec container ip link show eth0 | grep mtu
```

**Solutions:**

```bash
# Adjust MTU if needed
docker network create --opt "com.docker.network.driver.mtu"="1450" my-network

# Use host network for maximum performance
docker run --network host my-app

# Check for resource limits
docker update --cpus 2 --memory 4g container-name
```

### 8.4 Advanced Debugging with Netshoot

**Running Netshoot:**

```bash
# Attach to same network namespace as target container
docker run -it --rm \
  --network container:target-container \
  nicolaka/netshoot

# Or on same network
docker run -it --rm \
  --network my-network \
  nicolaka/netshoot
```

**Netshoot includes:**

- **Network tools**: ping, traceroute, nslookup, dig, host
- **Protocol tools**: curl, wget, httpie
- **Debugging**: tcpdump, iperf3, netstat, ss
- **Analysis**: mtr, nmap, socat
- **Utilities**: jq, yq, grpcurl

**Example Debugging Session:**

```bash
# Start netshoot
docker run -it --rm --network my-network nicolaka/netshoot

# Inside netshoot:

# 1. Check DNS
nslookup api-server
dig api-server

# 2. Test TCP connectivity
nc -zv api-server 5000

# 3. HTTP testing
curl -v http://api-server:5000/health
httpie http://api-server:5000/health

# 4. Capture packets
tcpdump -i eth0 -n -A port 5000

# 5. Performance test
iperf3 -c api-server

# 6. Route tracing
mtr api-server

# 7. Port scanning
nmap -p 1-10000 api-server
```

---

## Best Practices

### 9.1 Network Design Principles

**1. Use Custom Bridge Networks**

```yaml
# DON'T: Rely on default bridge
services:
  app:
    image: myapp
    # No network specified - uses default bridge
    # DNS won't work!

# DO: Use custom networks
services:
  app:
    image: myapp
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

**2. Implement Network Segmentation**

```yaml
# DON'T: Everything on one network
networks:
  default:
    driver: bridge

# DO: Separate by function/security tier
networks:
  public:      # Internet-facing
  application: # App logic
  data:        # Databases
    internal: true
```

**3. Use Descriptive Network Names**

```bash
# DON'T:
docker network create net1
docker network create net2

# DO:
docker network create ml-inference-network
docker network create ml-training-network
docker network create monitoring-network
```

**4. Minimize Exposed Ports**

```yaml
# DON'T: Expose everything
services:
  database:
    ports:
      - "5432:5432"  # Dangerous!

# DO: Use internal networking
services:
  database:
    # No ports section - only accessible from other containers
    networks:
      - backend
```

**5. Plan IP Ranges**

```bash
# DON'T: Let Docker assign random subnets
docker network create my-network

# DO: Use predictable, non-conflicting subnets
docker network create \
  --subnet 172.25.0.0/16 \
  --gateway 172.25.0.1 \
  ml-network
```

### 9.2 Security Best Practices

**1. Use Internal Networks for Sensitive Data**

```yaml
networks:
  data-tier:
    driver: bridge
    internal: true  # No external access
```

**2. Implement Network Policies**

```yaml
# Least privilege: only connect to necessary networks
services:
  web:
    networks:
      - frontend  # Only what's needed

  api:
    networks:
      - frontend
      - backend

  database:
    networks:
      - backend  # Not on frontend
```

**3. Avoid Host Network Mode in Production**

```yaml
# DON'T: (unless absolutely necessary)
services:
  app:
    network_mode: host

# DO: Use bridge with specific ports
services:
  app:
    ports:
      - "8080:8080"
    networks:
      - app-network
```

**4. Enable Network Encryption**

```bash
# For overlay networks (Swarm mode)
docker network create \
  --driver overlay \
  --opt encrypted \
  secure-overlay-net
```

**5. Regular Network Audits**

```bash
# List all networks
docker network ls

# Identify unused networks
docker network prune

# Review network configurations
for network in $(docker network ls -q); do
  echo "=== $network ==="
  docker network inspect $network | jq '.[].Name, .[].Driver, .[].Internal'
done
```

### 9.3 Performance Optimization

**1. Use Host Network for High-Throughput**

```bash
# For high-performance needs (monitoring, high-traffic inference)
docker run -d --network host high-throughput-app
```

**2. Configure MTU Appropriately**

```bash
# Match host's MTU to avoid fragmentation
docker network create \
  --opt "com.docker.network.driver.mtu"="1500" \
  optimized-network
```

**3. Tune Kernel Parameters**

```yaml
# /etc/docker/daemon.json
{
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
```

**4. Use Connection Pooling**

```python
# In application code
import psycopg2.pool

# Connection pool reduces network overhead
connection_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=10,
    maxconn=20,
    host='database',
    database='mldata'
)
```

### 9.4 ML-Specific Networking Best Practices

**1. Distributed Training Communication**

```yaml
# Use dedicated network with sufficient bandwidth
networks:
  training-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: ml-train
      com.docker.network.driver.mtu: "9000"  # Jumbo frames if supported
```

**2. Model Serving Patterns**

```yaml
# Separate inference from training networks
networks:
  inference-public:
    # External access for predictions
  inference-internal:
    internal: true
    # Model loading, feature access
```

**3. Data Pipeline Isolation**

```yaml
# Keep ETL separate from serving
networks:
  etl-network:
    internal: true
  serving-network:
    # Only inference services
```

**4. Monitoring Integration**

```yaml
# Monitoring can span networks
services:
  prometheus:
    networks:
      - inference-network
      - training-network
      - data-network
      - monitoring-network
```

**5. Feature Store Connectivity**

```yaml
# Online features: low-latency network
# Offline features: batch processing network
networks:
  online-features:
    driver: bridge
  offline-features:
    driver: bridge
    internal: true
```

### 9.5 Documentation and Naming Conventions

**Network Naming Convention:**

```
<environment>-<function>-network

Examples:
- prod-inference-network
- dev-training-network
- staging-data-network
```

**Network Diagram Example:**

```
┌─────────────────────────────────────────┐
│          Internet                       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│     public-network (bridge)             │
│  - Load Balancer                        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  inference-network (bridge, internal)   │
│  - Inference API                        │
│  - Model Server                         │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│    ml-network (bridge, internal)        │
│  - Feature Store                        │
│  - Model Cache                          │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│   data-network (bridge, internal)       │
│  - PostgreSQL                           │
│  - Redis                                │
└─────────────────────────────────────────┘
```

---

## Summary

### What You've Learned

1. **Network Types**: Bridge, host, overlay, macvlan, and none networks
2. **Container Communication**: DNS-based service discovery, network aliases
3. **Port Mapping**: Publishing ports, expose directive, host access
4. **Network Isolation**: Multi-tier architectures, internal networks, segmentation
5. **DNS Resolution**: How Docker DNS works, custom DNS configuration
6. **Load Balancing**: Nginx and HAProxy configurations, health checks
7. **ML Patterns**: Distributed training, model serving, feature stores
8. **Debugging**: Network inspection, connectivity testing, troubleshooting
9. **Best Practices**: Security, performance, naming conventions

### Key Takeaways

- **Always use custom bridge networks** for DNS resolution
- **Implement network segmentation** for security
- **Use internal networks** for sensitive data
- **Plan IP ranges** to avoid conflicts
- **Monitor network performance** in production
- **Document network architecture** for team understanding

### Next Steps

1. **Practice**: Implement the examples in this guide
2. **Experiment**: Try different network configurations
3. **Production**: Apply these patterns to real ML projects
4. **Advanced Topics**: Explore Kubernetes networking (next module)

---

**Exercise Completed Successfully!**

You now have comprehensive knowledge of Docker networking for ML infrastructure. Apply these concepts to build secure, scalable, and performant containerized ML systems.

**Next Exercise**: Volume Management and Data Persistence

---

**Version**: 1.0
**Last Updated**: November 2025
**Author**: AI Infrastructure Team
