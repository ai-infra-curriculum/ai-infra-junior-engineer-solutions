# Exercise 03: Debugging Kubernetes Applications

## Overview

This exercise provides hands-on practice with debugging common Kubernetes issues through real-world scenarios. You'll learn systematic troubleshooting approaches, essential kubectl commands, and how to diagnose and fix problems in production-like environments.

## Learning Objectives

By completing this exercise, you will:

1. Identify and debug common Kubernetes pod issues
2. Troubleshoot service connectivity problems
3. Diagnose resource constraint issues
4. Debug configuration and secret management
5. Fix health probe configurations
6. Use kubectl effectively for debugging
7. Analyze logs and events systematically
8. Apply structured troubleshooting methodologies

## Prerequisites

- Completed Exercise 01 (First Kubernetes Deployment)
- Completed Exercise 02 (Helm Chart)
- Kubernetes cluster (kind, minikube, or cloud provider)
- kubectl configured and working
- Basic understanding of:
  - Kubernetes resources (Pods, Services, Deployments)
  - kubectl commands
  - YAML syntax
  - Linux command line

## Exercise Structure

This exercise contains **6 debugging scenarios** covering the most common Kubernetes issues:

| Scenario | Issue Type | Difficulty | Key Learning |
|----------|-----------|------------|--------------|
| 01 | Image Pull Error | Beginner | Image configuration, registry issues |
| 02 | CrashLoopBackOff | Beginner | Application crashes, log analysis |
| 03 | Resource Constraints | Intermediate | Resource management, scheduling |
| 04 | Service Connectivity | Intermediate | Service discovery, labels, endpoints |
| 05 | Configuration Issues | Intermediate | ConfigMaps, Secrets, volume mounts |
| 06 | Probe Failures | Advanced | Health checks, probe configuration |

## Directory Structure

```
exercise-03-debugging/
├── README.md                          # This file
├── STEP_BY_STEP.md                    # Detailed walkthrough
├── scenarios/                         # Broken deployment scenarios
│   ├── 01-image-pull-error.yaml
│   ├── 02-crashloop-backoff.yaml
│   ├── 03-resource-constraints.yaml
│   ├── 04-service-connectivity.yaml
│   ├── 05-config-issues.yaml
│   └── 06-liveness-readiness.yaml
├── scripts/                           # Automation and debugging tools
│   ├── debug-master.sh               # Comprehensive debugging script
│   ├── deploy-scenarios.sh           # Deploy practice scenarios
│   └── cleanup-scenarios.sh          # Clean up scenarios
└── docs/
    └── TROUBLESHOOTING_GUIDE.md      # Quick reference guide
```

## Quick Start

### 1. Deploy All Scenarios

```bash
cd scripts
./deploy-scenarios.sh all
```

This deploys all 6 scenarios, each in its own namespace (`debug-scenario-01` through `debug-scenario-06`).

### 2. Choose a Scenario

Start with Scenario 01 (easiest):

```bash
# Check the deployed resources
kubectl get all -n debug-scenario-01

# Check pod status (you'll see the problem)
kubectl get pods -n debug-scenario-01
```

### 3. Debug the Issue

Use the master debugging script:

```bash
./debug-master.sh -n debug-scenario-01 check-all
```

Or use kubectl commands directly:

```bash
# Describe the pod to see events
kubectl describe pod -n debug-scenario-01 <pod-name>

# Check events
kubectl get events -n debug-scenario-01 --sort-by='.lastTimestamp'
```

### 4. Fix the Issue

Each scenario file contains debugging hints in comments. Try to fix it yourself first, then refer to the hints if needed.

### 5. Verify the Fix

```bash
# Check if pods are now running
kubectl get pods -n debug-scenario-01

# Verify all resources are healthy
./debug-master.sh -n debug-scenario-01 check-all
```

### 6. Clean Up

```bash
./cleanup-scenarios.sh 01

# Or clean up all scenarios
./cleanup-scenarios.sh all
```

## Detailed Scenario Descriptions

### Scenario 01: Image Pull Error

**Problem**: Deployment fails because the container image name contains a typo.

**Symptoms**:
- Pod status: `ImagePullBackOff` or `ErrImagePull`
- Events show "Failed to pull image"
- Pods remain in Pending state

**Learning Focus**:
- How to identify image pull failures
- Reading pod events
- Fixing image specifications
- Understanding image naming conventions

**Key Commands**:
```bash
kubectl describe pod -n debug-scenario-01 <pod-name>
kubectl get events -n debug-scenario-01
kubectl set image deployment/broken-nginx nginx=nginx:1.21-alpine -n debug-scenario-01
```

---

### Scenario 02: CrashLoopBackOff

**Problem**: Application crashes immediately after startup due to invalid configuration file.

**Symptoms**:
- Pod status: `CrashLoopBackOff`
- High restart count
- Container exits shortly after starting

**Learning Focus**:
- Analyzing container logs
- Understanding exit codes
- Debugging application startup failures
- Fixing ConfigMap issues

**Key Commands**:
```bash
kubectl logs -n debug-scenario-02 <pod-name>
kubectl logs -n debug-scenario-02 <pod-name> --previous
kubectl get configmap broken-app-config -n debug-scenario-02 -o yaml
kubectl edit configmap broken-app-config -n debug-scenario-02
```

---

### Scenario 03: Resource Constraints

**Problem**: Multiple resource-related issues including insufficient resources, missing limits, and OOMKilled containers.

**Symptoms**:
- Pods in `Pending` state (insufficient resources)
- Pods with `OOMKilled` status (out of memory)
- Pods without resource limits (bad practice)

**Learning Focus**:
- Understanding resource requests and limits
- Debugging scheduling failures
- Analyzing memory usage
- Setting appropriate resource constraints

**Key Commands**:
```bash
kubectl describe pod -n debug-scenario-03 <pod-name>
kubectl top nodes
kubectl top pods -n debug-scenario-03
kubectl set resources deployment/resource-hungry-app --requests=cpu=100m,memory=128Mi -n debug-scenario-03
```

---

### Scenario 04: Service Connectivity

**Problem**: Services not routing traffic to pods due to incorrect selectors and port misconfigurations.

**Symptoms**:
- Service has no endpoints
- Connection refused errors
- Cannot reach application through service

**Learning Focus**:
- Understanding service selectors
- Debugging endpoint issues
- Testing service connectivity
- Matching labels correctly

**Key Commands**:
```bash
kubectl get endpoints -n debug-scenario-04
kubectl describe svc backend-service-broken -n debug-scenario-04
kubectl get pods -n debug-scenario-04 --show-labels
kubectl exec -n debug-scenario-04 client-test-pod -- curl http://backend-service-correct
```

---

### Scenario 05: Configuration Issues

**Problem**: Multiple configuration-related problems including missing ConfigMaps, wrong Secret keys, and volume mount issues.

**Symptoms**:
- Pod status: `CreateContainerConfigError`
- Pods not starting
- Missing environment variables

**Learning Focus**:
- Debugging ConfigMap issues
- Validating Secret references
- Checking volume mounts
- Understanding configuration injection

**Key Commands**:
```bash
kubectl describe pod -n debug-scenario-05 <pod-name>
kubectl get configmap -n debug-scenario-05
kubectl get secret app-secrets -n debug-scenario-05 -o yaml
kubectl create configmap nonexistent-config --from-literal=app.env=production -n debug-scenario-05
```

---

### Scenario 06: Liveness and Readiness Probes

**Problem**: Incorrect health probe configurations causing pods to restart frequently or not receive traffic.

**Symptoms**:
- Pods restarting frequently
- Service not routing traffic to ready pods
- Events show probe failures

**Learning Focus**:
- Understanding liveness vs readiness probes
- Configuring probes correctly
- Debugging probe failures
- Setting appropriate timeouts and thresholds

**Key Commands**:
```bash
kubectl get pods -n debug-scenario-06 -w
kubectl describe pod -n debug-scenario-06 <pod-name> | grep -A 5 "probe failed"
kubectl get endpoints -n debug-scenario-06
kubectl patch deployment broken-liveness-app -n debug-scenario-06 --type=json -p='[...]'
```

---

## Debugging Tools and Scripts

### debug-master.sh

Comprehensive debugging script with multiple check modes:

```bash
# Run all diagnostic checks
./debug-master.sh -n <namespace> check-all

# Check specific components
./debug-master.sh -n <namespace> check-pods
./debug-master.sh -n <namespace> check-services
./debug-master.sh -n <namespace> check-network
./debug-master.sh -n <namespace> check-resources
./debug-master.sh -n <namespace> check-config
./debug-master.sh -n <namespace> check-logs -p <pod-name>
./debug-master.sh -n <namespace> check-events

# Interactive mode
./debug-master.sh -n <namespace> interactive
```

**Features**:
- Automated diagnostics for common issues
- Color-coded output for easy reading
- Detailed analysis and recommendations
- Interactive debugging mode
- Pod, service, and network checks
- Resource usage analysis
- Configuration validation
- Log and event analysis

### deploy-scenarios.sh

Deploy debugging scenarios for practice:

```bash
# Interactive mode
./deploy-scenarios.sh

# Deploy specific scenario
./deploy-scenarios.sh 01

# Deploy all scenarios
./deploy-scenarios.sh all
```

### cleanup-scenarios.sh

Clean up deployed scenarios:

```bash
# Interactive mode
./cleanup-scenarios.sh

# Clean up specific scenario
./cleanup-scenarios.sh 01

# Clean up all scenarios
./cleanup-scenarios.sh all
```

---

## Systematic Debugging Approach

When debugging Kubernetes issues, follow this systematic approach:

### 1. Identify the Problem

```bash
# Start with high-level view
kubectl get all -n <namespace>

# Check pod status
kubectl get pods -n <namespace> -o wide

# Look for obvious issues (Pending, CrashLoopBackOff, Error, etc.)
```

### 2. Gather Information

```bash
# Describe the problematic resource
kubectl describe pod <pod-name> -n <namespace>

# Check recent events
kubectl get events -n <namespace> --sort-by='.lastTimestamp' | tail -20

# View logs
kubectl logs <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace> --previous  # For crashed containers
```

### 3. Analyze Root Cause

Look for patterns in:
- **Events**: Indicate what went wrong and when
- **Logs**: Application-level errors and stack traces
- **Status**: Pod phase, container statuses, conditions
- **Configuration**: Environment variables, volumes, probes

### 4. Formulate Hypothesis

Based on symptoms and evidence, determine likely cause:
- Image issues? → Check image name, tag, pull policy
- Crash issues? → Check logs, exit code, application errors
- Resource issues? → Check node capacity, resource requests/limits
- Network issues? → Check service selectors, endpoints, DNS
- Config issues? → Check ConfigMaps, Secrets, environment variables
- Probe issues? → Check probe configuration, endpoints, timeouts

### 5. Test and Fix

```bash
# Make targeted fix
kubectl edit <resource> -n <namespace>
# OR
kubectl patch <resource> <name> -n <namespace> -p '<patch>'
# OR
kubectl set <command> -n <namespace>

# Verify fix
kubectl get pods -n <namespace> -w
kubectl describe pod <pod-name> -n <namespace>
```

### 6. Verify Solution

```bash
# Ensure pods are running and ready
kubectl get pods -n <namespace>

# Verify no recent errors in events
kubectl get events -n <namespace> --field-selector type=Warning

# Test functionality
kubectl exec <pod-name> -n <namespace> -- <test-command>
```

---

## Essential kubectl Commands for Debugging

### Pod Debugging

```bash
# Get pod status
kubectl get pods -n <namespace>
kubectl get pod <pod-name> -n <namespace> -o wide
kubectl get pod <pod-name> -n <namespace> -o yaml

# Describe pod (most useful!)
kubectl describe pod <pod-name> -n <namespace>

# View logs
kubectl logs <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace> -c <container-name>  # Multi-container
kubectl logs <pod-name> -n <namespace> --previous            # Previous instance
kubectl logs <pod-name> -n <namespace> -f                    # Follow logs
kubectl logs <pod-name> -n <namespace> --tail=50             # Last 50 lines

# Execute commands in pod
kubectl exec <pod-name> -n <namespace> -- <command>
kubectl exec -it <pod-name> -n <namespace> -- /bin/sh

# Port forward for local testing
kubectl port-forward <pod-name> -n <namespace> 8080:80

# Copy files to/from pod
kubectl cp <pod-name>:/path/to/file ./local-file -n <namespace>
kubectl cp ./local-file <pod-name>:/path/to/file -n <namespace>
```

### Service Debugging

```bash
# Get service and endpoints
kubectl get svc -n <namespace>
kubectl get endpoints -n <namespace>
kubectl describe svc <service-name> -n <namespace>

# Check service selector
kubectl get svc <service-name> -n <namespace> -o jsonpath='{.spec.selector}'

# Find pods matching service selector
kubectl get pods -n <namespace> -l <label-key>=<label-value>
```

### Event Debugging

```bash
# View all events
kubectl get events -n <namespace>

# Sort by timestamp
kubectl get events -n <namespace> --sort-by='.lastTimestamp'

# Filter by type
kubectl get events -n <namespace> --field-selector type=Warning

# Filter by involved object
kubectl get events -n <namespace> --field-selector involvedObject.name=<pod-name>
```

### Resource Usage

```bash
# Node resources
kubectl top nodes
kubectl describe nodes

# Pod resources
kubectl top pods -n <namespace>
kubectl top pod <pod-name> -n <namespace>

# Check resource quotas
kubectl get resourcequota -n <namespace>
kubectl describe resourcequota -n <namespace>
```

### Configuration Debugging

```bash
# ConfigMaps
kubectl get configmap -n <namespace>
kubectl describe configmap <name> -n <namespace>
kubectl get configmap <name> -n <namespace> -o yaml

# Secrets
kubectl get secret -n <namespace>
kubectl describe secret <name> -n <namespace>
kubectl get secret <name> -n <namespace> -o jsonpath='{.data}'

# Decode secret
kubectl get secret <name> -n <namespace> -o jsonpath='{.data.<key>}' | base64 -d
```

---

## Tips and Best Practices

### Debugging Tips

1. **Always check events first**: Use `kubectl get events` to quickly identify issues
2. **Use `--previous` for crashed containers**: Current logs may be empty
3. **Describe before digging deeper**: `kubectl describe` often reveals the problem
4. **Check labels and selectors**: Mismatched labels are very common
5. **Verify resource limits**: Too low limits cause OOMKills
6. **Test from inside the cluster**: Use a debug pod to test connectivity
7. **Use `-o yaml` for full resource details**: Hidden fields may contain clues
8. **Check recent changes**: Issues often correlate with recent deployments

### Common Mistakes to Avoid

1. **Not checking events**: Events are the first place to look
2. **Ignoring resource requests/limits**: Causes scheduling and performance issues
3. **Mismatched labels**: Service selectors must match pod labels exactly
4. **Wrong port numbers**: targetPort must match container port
5. **Case sensitivity**: Label values are case-sensitive
6. **Probe configuration**: Too aggressive probes cause unnecessary restarts
7. **Not using `--previous`**: Crashed container logs need `--previous` flag
8. **Assuming DNS works**: Always test DNS resolution in debug scenarios

---

## Troubleshooting Workflow

```
┌─────────────────────────────────────┐
│  Problem Detected                    │
│  (Pod not running, service down)     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Check Pod Status                    │
│  kubectl get pods -n <namespace>     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Describe Pod                        │
│  kubectl describe pod <name>         │
│  • Check Events                      │
│  • Check Conditions                  │
│  • Check Container Status            │
└────────────┬────────────────────────┘
             │
     ┌───────┴───────┐
     │               │
     ▼               ▼
┌─────────┐    ┌──────────┐
│  Events  │    │   Logs   │
│  Issues? │    │  Issues? │
└────┬─────┘    └────┬─────┘
     │               │
     └───────┬───────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Identify Root Cause                 │
│  • Image issues?                     │
│  • Resource constraints?             │
│  • Configuration errors?             │
│  • Network problems?                 │
│  • Probe failures?                   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Apply Fix                           │
│  kubectl patch/edit/set              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Verify Fix                          │
│  • Pods running?                     │
│  • Events clean?                     │
│  • Application working?              │
└─────────────────────────────────────┘
```

---

## Additional Resources

### Documentation
- [Kubernetes Debugging Docs](https://kubernetes.io/docs/tasks/debug/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Troubleshooting Guide](./docs/TROUBLESHOOTING_GUIDE.md)

### Helpful Tools
- **stern**: Multi-pod log tailing
- **k9s**: Terminal UI for Kubernetes
- **kubectl-debug**: Ephemeral debug containers
- **kubectx/kubens**: Fast context switching
- **krew**: kubectl plugin manager

### Learning Resources
- [Step-by-Step Guide](./STEP_BY_STEP.md)
- [Kubernetes Events Documentation](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#event-v1-core)

---

## Next Steps

After completing this exercise, you should:

1. Be comfortable using kubectl for debugging
2. Understand common Kubernetes issues
3. Know how to read logs and events effectively
4. Be able to diagnose and fix pod/service issues
5. Have a systematic approach to troubleshooting

**Continue to**: Exercise 04 - StatefulSets and Persistent Storage

---

## Support

For questions or issues with this exercise:
- Review the [TROUBLESHOOTING_GUIDE.md](./docs/TROUBLESHOOTING_GUIDE.md)
- Check the [STEP_BY_STEP.md](./STEP_BY_STEP.md) guide
- Examine comments in scenario files
- Consult Kubernetes debugging documentation

---

**Exercise Type**: Hands-on Debugging Practice
**Difficulty**: Beginner to Intermediate
**Estimated Time**: 3-4 hours
**Prerequisites**: Exercise 01, Exercise 02, Basic kubectl knowledge
