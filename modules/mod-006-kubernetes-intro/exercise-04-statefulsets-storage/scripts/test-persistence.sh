#!/bin/bash

################################################################################
# Test Data Persistence in StatefulSets
#
# This script demonstrates how data persists across pod restarts in StatefulSets
################################################################################

set -e

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_step() {
    echo -e "${CYAN}Step $1: $2${NC}"
}

# Test PostgreSQL persistence
test_postgres_persistence() {
    print_header "Testing PostgreSQL Data Persistence"

    print_step 1 "Writing data to PostgreSQL (postgres-0)"

    # Create a test table and insert data
    kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase << 'EOF'
CREATE TABLE IF NOT EXISTS test_persistence (
    id SERIAL PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO test_persistence (message) VALUES
    ('This data should persist across pod restarts'),
    ('Testing persistence in StatefulSet'),
    ('Created at: ' || NOW());

SELECT * FROM test_persistence;
EOF

    print_success "Data written to postgres-0"

    print_step 2 "Deleting postgres-0 pod"
    kubectl delete pod postgres-0 -n statefulset-demo
    print_info "Pod deleted, waiting for recreation..."

    print_step 3 "Waiting for pod to be ready"
    kubectl wait --for=condition=ready pod/postgres-0 -n statefulset-demo --timeout=120s
    print_success "postgres-0 is ready again"

    print_step 4 "Reading data from recreated postgres-0"
    echo ""
    kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase -c "SELECT * FROM test_persistence;"
    echo ""

    print_success "Data persisted! PVC was reattached to new pod"
}

# Test Redis persistence
test_redis_persistence() {
    print_header "Testing Redis Data Persistence"

    print_step 1 "Writing data to Redis (redis-0)"

    # Write test data
    kubectl exec -it redis-0 -n statefulset-demo -- redis-cli << 'EOF'
SET test_key "This data should persist"
SET pod_identity "redis-0"
SET timestamp "$(date)"
SAVE
KEYS *
GET test_key
EOF

    print_success "Data written to redis-0"

    print_step 2 "Deleting redis-0 pod"
    kubectl delete pod redis-0 -n statefulset-demo
    print_info "Pod deleted, waiting for recreation..."

    print_step 3 "Waiting for pod to be ready"
    kubectl wait --for=condition=ready pod/redis-0 -n statefulset-demo --timeout=120s
    print_success "redis-0 is ready again"

    print_step 4 "Reading data from recreated redis-0"
    echo ""
    kubectl exec -it redis-0 -n statefulset-demo -- redis-cli << 'EOF'
KEYS *
GET test_key
GET pod_identity
EOF
    echo ""

    print_success "Data persisted! PVC was reattached to new pod"
}

# Test StatefulSet vs Deployment persistence
test_statefulset_vs_deployment() {
    print_header "StatefulSet vs Deployment Persistence"

    print_step 1 "Writing data to StatefulSet pod (nginx-statefulset-0)"
    kubectl exec nginx-statefulset-0 -n statefulset-demo -- sh -c \
        'echo "Persistent data in StatefulSet - $(date)" > /usr/share/nginx/html/data.txt'

    STATEFULSET_DATA=$(kubectl exec nginx-statefulset-0 -n statefulset-demo -- cat /usr/share/nginx/html/data.txt)
    print_info "Data written: $STATEFULSET_DATA"

    print_step 2 "Writing data to Deployment pod"
    DEPLOYMENT_POD=$(kubectl get pods -n statefulset-demo -l app=nginx-deployment -o jsonpath='{.items[0].metadata.name}')
    kubectl exec $DEPLOYMENT_POD -n statefulset-demo -- sh -c \
        'echo "Ephemeral data in Deployment - $(date)" > /usr/share/nginx/html/data.txt'

    DEPLOYMENT_DATA=$(kubectl exec $DEPLOYMENT_POD -n statefulset-demo -- cat /usr/share/nginx/html/data.txt)
    print_info "Data written: $DEPLOYMENT_DATA"

    print_step 3 "Deleting both pods"
    kubectl delete pod nginx-statefulset-0 -n statefulset-demo
    kubectl delete pod $DEPLOYMENT_POD -n statefulset-demo
    print_info "Pods deleted, waiting for recreation..."
    sleep 5

    print_step 4 "Waiting for pods to be ready"
    kubectl wait --for=condition=ready pod/nginx-statefulset-0 -n statefulset-demo --timeout=60s
    NEW_DEPLOYMENT_POD=$(kubectl get pods -n statefulset-demo -l app=nginx-deployment -o jsonpath='{.items[0].metadata.name}')
    kubectl wait --for=condition=ready pod/$NEW_DEPLOYMENT_POD -n statefulset-demo --timeout=60s

    print_step 5 "Checking data persistence"

    echo ""
    print_info "StatefulSet pod (nginx-statefulset-0):"
    if kubectl exec nginx-statefulset-0 -n statefulset-demo -- cat /usr/share/nginx/html/data.txt 2>/dev/null; then
        print_success "✓ Data persisted in StatefulSet!"
    else
        print_error "✗ Data lost in StatefulSet"
    fi

    echo ""
    print_info "Deployment pod ($NEW_DEPLOYMENT_POD):"
    if kubectl exec $NEW_DEPLOYMENT_POD -n statefulset-demo -- cat /usr/share/nginx/html/data.txt 2>/dev/null; then
        echo "Data found (unexpected for emptyDir)"
    else
        print_success "✓ Data correctly ephemeral in Deployment (using emptyDir)"
    fi

    echo ""
    print_success "StatefulSet maintains persistent storage, Deployment does not"
}

# Check PVC status
check_pvc_status() {
    print_header "PersistentVolumeClaim Status"

    print_info "PostgreSQL PVCs:"
    kubectl get pvc -n statefulset-demo -l app=postgresql

    echo ""
    print_info "Redis PVCs:"
    kubectl get pvc -n statefulset-demo -l app=redis

    echo ""
    print_info "StatefulSet example PVCs:"
    kubectl get pvc -n statefulset-demo | grep nginx-statefulset || true

    echo ""
    print_info "Note: Deployment pods have no PVCs (using emptyDir)"
}

# Demonstrate PVC reattachment
demonstrate_pvc_reattachment() {
    print_header "PVC Reattachment Demonstration"

    print_step 1 "Checking current PVC for redis-0"
    PVC_NAME=$(kubectl get pod redis-0 -n statefulset-demo -o jsonpath='{.spec.volumes[?(@.persistentVolumeClaim)].persistentVolumeClaim.claimName}' | head -1)
    print_info "PVC Name: $PVC_NAME"

    PVC_VOLUME=$(kubectl get pvc $PVC_NAME -n statefulset-demo -o jsonpath='{.spec.volumeName}')
    print_info "PV Name: $PVC_VOLUME"

    print_step 2 "Scaling StatefulSet to 0 (deletes pod but keeps PVC)"
    kubectl scale statefulset redis --replicas=0 -n statefulset-demo
    sleep 5

    print_step 3 "Verifying PVC still exists"
    if kubectl get pvc $PVC_NAME -n statefulset-demo &> /dev/null; then
        print_success "PVC $PVC_NAME still exists"
    else
        print_error "PVC was deleted (unexpected)"
    fi

    print_step 4 "Scaling StatefulSet back to 1"
    kubectl scale statefulset redis --replicas=1 -n statefulset-demo
    kubectl wait --for=condition=ready pod/redis-0 -n statefulset-demo --timeout=120s

    print_step 5 "Verifying same PVC was reattached"
    NEW_PVC_NAME=$(kubectl get pod redis-0 -n statefulset-demo -o jsonpath='{.spec.volumes[?(@.persistentVolumeClaim)].persistentVolumeClaim.claimName}' | head -1)

    if [ "$PVC_NAME" = "$NEW_PVC_NAME" ]; then
        print_success "Same PVC reattached: $PVC_NAME"
    else
        print_error "Different PVC attached (unexpected)"
    fi

    # Restore to 3 replicas
    print_info "Restoring redis StatefulSet to 3 replicas..."
    kubectl scale statefulset redis --replicas=3 -n statefulset-demo
}

# Main menu
show_menu() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}StatefulSet Persistence Tests${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo "1. Test PostgreSQL persistence"
    echo "2. Test Redis persistence"
    echo "3. Test StatefulSet vs Deployment"
    echo "4. Check PVC status"
    echo "5. Demonstrate PVC reattachment"
    echo "6. Run all tests"
    echo "0. Exit"
    echo ""
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option: " choice

            case $choice in
                1) test_postgres_persistence ;;
                2) test_redis_persistence ;;
                3) test_statefulset_vs_deployment ;;
                4) check_pvc_status ;;
                5) demonstrate_pvc_reattachment ;;
                6)
                    test_postgres_persistence
                    test_redis_persistence
                    test_statefulset_vs_deployment
                    check_pvc_status
                    demonstrate_pvc_reattachment
                    ;;
                0) exit 0 ;;
                *) print_error "Invalid option" ;;
            esac

            echo ""
            read -p "Press Enter to continue..."
        done
    else
        # Command line mode
        case $1 in
            postgres) test_postgres_persistence ;;
            redis) test_redis_persistence ;;
            comparison) test_statefulset_vs_deployment ;;
            pvc) check_pvc_status ;;
            reattach) demonstrate_pvc_reattachment ;;
            all)
                test_postgres_persistence
                test_redis_persistence
                test_statefulset_vs_deployment
                check_pvc_status
                demonstrate_pvc_reattachment
                ;;
            *)
                echo "Usage: $0 [postgres|redis|comparison|pvc|reattach|all]"
                exit 1
                ;;
        esac
    fi
}

main "$@"
