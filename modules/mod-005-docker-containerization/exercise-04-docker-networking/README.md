# Exercise 04: Docker Networking

Master Docker networking for ML applications including custom networks, service discovery, and network security.

## Key Concepts
- Bridge networks (default)
- Host networks (better performance)
- Custom networks
- Service discovery
- Port mapping

## Solution
See `solutions/network_manager.sh` for automated network management.

## Quick Examples

```bash
# Create custom network
docker network create ml-network

# Run containers on network
docker run -d --name api --network ml-network my-api
docker run -d --name db --network ml-network postgres

# Containers can communicate via service names
# api can reach db at hostname "db"
```
