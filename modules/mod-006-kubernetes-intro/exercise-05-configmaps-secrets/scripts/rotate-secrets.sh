#!/bin/bash

# rotate-secrets.sh
# Demonstrates secret rotation patterns in Kubernetes
# Shows both manual and automated secret rotation strategies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="config-demo"
SECRET_NAME="app-secrets"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# Function to check if secret exists
check_secret_exists() {
    local secret=$1
    if kubectl get secret "$secret" -n "$NAMESPACE" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to backup current secret
backup_secret() {
    local secret=$1
    local backup_name="${secret}-backup-${TIMESTAMP}"

    print_info "Backing up secret '$secret' to '$backup_name'..."

    # Export current secret
    kubectl get secret "$secret" -n "$NAMESPACE" -o yaml | \
        sed "s/name: $secret/name: $backup_name/" | \
        kubectl apply -f - > /dev/null

    print_success "Secret backed up to '$backup_name'"
    echo "  Backup name: $backup_name"
}

# Function to generate new password
generate_password() {
    local length=${1:-20}
    # Generate random password using openssl
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-${length}
}

# Function to rotate specific secret key
rotate_secret_key() {
    local secret=$1
    local key=$2
    local new_value=$3

    print_info "Rotating key '$key' in secret '$secret'..."

    # Get current secret
    local current_secret=$(kubectl get secret "$secret" -n "$NAMESPACE" -o json)

    # Update the specific key
    local encoded_value=$(echo -n "$new_value" | base64)

    # Patch the secret
    kubectl patch secret "$secret" -n "$NAMESPACE" --type='json' \
        -p='[{"op": "replace", "path": "/data/'$key'", "value": "'$encoded_value'"}]' > /dev/null

    print_success "Key '$key' rotated successfully"
}

# Pattern 1: Zero-Downtime Secret Rotation with Dual Secrets
rotate_with_dual_secrets() {
    print_header "Pattern 1: Zero-Downtime Rotation (Dual Secrets)"

    print_info "This pattern uses two secrets (current + new) for graceful rotation"
    echo ""

    local secret_current="app-secrets-current"
    local secret_new="app-secrets-new"

    print_info "Step 1: Create new secret with rotated credentials"
    local new_password=$(generate_password)

    kubectl create secret generic "$secret_new" \
        --from-literal=database-password="$new_password" \
        --from-literal=api-key="$(generate_password)" \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f - > /dev/null

    print_success "New secret created: $secret_new"
    echo ""

    print_info "Step 2: Update application to use both secrets (gradual rollout)"
    echo "  - Application reads from both current and new secret"
    echo "  - New connections use new credentials"
    echo "  - Existing connections continue with old credentials"
    echo ""

    print_info "Step 3: Wait for all connections to migrate to new credentials"
    echo "  - Monitor application logs"
    echo "  - Ensure all pods are using new secret"
    print_warning "In production, wait for 2x max connection lifetime"
    echo ""

    print_info "Step 4: Remove old secret and rename new -> current"
    echo "  kubectl delete secret $secret_current -n $NAMESPACE"
    echo "  kubectl create secret generic $secret_current --from-literal=... -n $NAMESPACE"
    echo ""

    print_success "Dual secret rotation complete (simulation)"
}

# Pattern 2: Rolling Restart Secret Rotation
rotate_with_rolling_restart() {
    print_header "Pattern 2: Rolling Restart Secret Rotation"

    print_info "This pattern updates secret and triggers rolling restart"
    echo ""

    if ! check_secret_exists "$SECRET_NAME"; then
        print_warning "Secret '$SECRET_NAME' not found, skipping..."
        return
    fi

    # Backup current secret
    backup_secret "$SECRET_NAME"
    echo ""

    print_info "Step 1: Update secret values"
    local new_db_password=$(generate_password)
    local new_api_key=$(generate_password)

    rotate_secret_key "$SECRET_NAME" "database-password" "$new_db_password"
    rotate_secret_key "$SECRET_NAME" "api-key" "$new_api_key"
    echo ""

    print_info "Step 2: Trigger rolling restart of deployments"
    print_warning "Note: This causes brief service disruption per pod"

    # Find deployments using this secret
    local deployments=$(kubectl get deployments -n "$NAMESPACE" -o json | \
        jq -r '.items[] | select(.spec.template.spec.volumes[]?.secret.secretName == "'$SECRET_NAME'") | .metadata.name' 2>/dev/null)

    if [ -n "$deployments" ]; then
        echo "  Deployments using this secret:"
        echo "$deployments" | while read -r deployment; do
            echo "    - $deployment"
            print_info "Restarting deployment: $deployment"
            kubectl rollout restart deployment "$deployment" -n "$NAMESPACE" > /dev/null
        done
        echo ""
        print_success "Rolling restarts triggered"
    else
        print_info "No deployments found using this secret"
    fi
}

# Pattern 3: Immutable Secret Rotation
rotate_with_immutable_secrets() {
    print_header "Pattern 3: Immutable Secret Rotation"

    print_info "This pattern creates new immutable secret with version suffix"
    echo ""

    local base_name="app-secrets-versioned"
    local version="v$(date +%Y%m%d%H%M%S)"
    local secret_name="${base_name}-${version}"

    print_info "Step 1: Create new immutable secret with version: $version"

    kubectl create secret generic "$secret_name" \
        --from-literal=database-password="$(generate_password)" \
        --from-literal=api-key="$(generate_password)" \
        --from-literal=jwt-secret="$(generate_password)" \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | \
        kubectl apply -f - > /dev/null

    # Make it immutable
    kubectl patch secret "$secret_name" -n "$NAMESPACE" \
        --type='merge' -p='{"immutable": true}' > /dev/null

    print_success "Immutable secret created: $secret_name"
    echo ""

    print_info "Step 2: Update deployments to reference new secret"
    echo "  Example:"
    echo "    kubectl set env deployment/myapp -n $NAMESPACE \\"
    echo "      --from=secret/$secret_name"
    echo ""

    print_info "Step 3: Clean up old versioned secrets after migration"
    local old_secrets=$(kubectl get secrets -n "$NAMESPACE" -o name | \
        grep "secret/${base_name}-v" | grep -v "$secret_name" | head -3)

    if [ -n "$old_secrets" ]; then
        echo "  Old secrets that could be deleted:"
        echo "$old_secrets"
    fi
    echo ""

    print_success "Immutable secret rotation complete"
}

# Pattern 4: External Secrets Operator Pattern
rotate_with_external_secrets() {
    print_header "Pattern 4: External Secrets Operator (Simulation)"

    print_info "This pattern uses external secret management systems"
    echo ""

    print_info "Typical flow with HashiCorp Vault:"
    echo "  1. Update secret in Vault"
    echo "     vault kv put secret/myapp/database password=$(generate_password)"
    echo ""
    echo "  2. External Secrets Operator detects change"
    echo "     - Polls Vault every N seconds (refreshInterval)"
    echo "     - Compares hash of Vault secret with K8s secret"
    echo ""
    echo "  3. Operator updates Kubernetes secret automatically"
    echo "     - Creates new secret or updates existing"
    echo "     - Pods using volume mounts see update after kubelet sync (~60s)"
    echo "     - Pods using env vars need restart"
    echo ""
    echo "  4. Optional: Reloader triggers rolling restart"
    echo "     - Stakater Reloader watches secrets"
    echo "     - Automatically restarts deployments when secret changes"
    echo ""

    print_info "Example ExternalSecret CRD:"
    cat <<EOF

apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: app-secrets-external
  namespace: $NAMESPACE
spec:
  refreshInterval: 15s  # Check Vault every 15 seconds
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: app-secrets  # K8s secret to create
    creationPolicy: Owner
  data:
  - secretKey: database-password
    remoteRef:
      key: secret/myapp/database
      property: password
  - secretKey: api-key
    remoteRef:
      key: secret/myapp/api
      property: key
EOF

    echo ""
    print_success "External Secrets pattern explained"
}

# Pattern 5: Blue-Green Secret Rotation
rotate_with_blue_green() {
    print_header "Pattern 5: Blue-Green Deployment Secret Rotation"

    print_info "This pattern uses blue-green deployments for secret rotation"
    echo ""

    local secret_blue="app-secrets-blue"
    local secret_green="app-secrets-green"

    print_info "Step 1: Create green secret with new credentials"
    kubectl create secret generic "$secret_green" \
        --from-literal=database-password="$(generate_password)" \
        --from-literal=api-key="$(generate_password)" \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f - > /dev/null

    print_success "Green secret created"
    echo ""

    print_info "Step 2: Deploy green environment using new secret"
    echo "  kubectl apply -f deployment-green.yaml"
    echo ""

    print_info "Step 3: Test green environment"
    echo "  - Run smoke tests"
    echo "  - Validate connectivity with new credentials"
    echo "  - Check logs for errors"
    echo ""

    print_info "Step 4: Switch traffic from blue to green"
    echo "  kubectl patch service myapp -n $NAMESPACE \\"
    echo "    -p '{\"spec\":{\"selector\":{\"version\":\"green\"}}}'"
    echo ""

    print_info "Step 5: Monitor and optionally rollback"
    echo "  - If issues detected, switch back to blue"
    echo "  - If successful, scale down blue deployment"
    echo ""

    print_success "Blue-green rotation pattern explained"
}

# Function to demonstrate secret rotation best practices
show_best_practices() {
    print_header "Secret Rotation Best Practices"

    echo "1. Always backup secrets before rotation"
    echo "   - Use version control for secret manifests (encrypted)"
    echo "   - Keep audit trail of rotation events"
    echo ""

    echo "2. Use automated rotation when possible"
    echo "   - External Secrets Operator + Vault"
    echo "   - AWS Secrets Manager with rotation Lambda"
    echo "   - Azure Key Vault with rotation policy"
    echo ""

    echo "3. Implement graceful degradation"
    echo "   - Accept both old and new credentials during transition"
    echo "   - Set overlap period for credential validity"
    echo ""

    echo "4. Monitor rotation process"
    echo "   - Track authentication failures"
    echo "   - Alert on rotation anomalies"
    echo "   - Log all rotation events"
    echo ""

    echo "5. Test rotation in non-production first"
    echo "   - Automate rotation testing"
    echo "   - Verify application recovery"
    echo ""

    echo "6. Set rotation schedules based on risk"
    echo "   - High-risk: Every 30-90 days"
    echo "   - Medium-risk: Every 6 months"
    echo "   - After any suspected compromise: Immediately"
    echo ""

    echo "7. Use immutable secrets when possible"
    echo "   - Prevents accidental overwrites"
    echo "   - Provides clear versioning"
    echo "   - Easier rollback"
    echo ""
}

# Function to verify rotation
verify_rotation() {
    print_header "Verifying Secret Rotation"

    local secret=$1

    if ! check_secret_exists "$secret"; then
        print_error "Secret '$secret' not found"
        return 1
    fi

    print_info "Secret: $secret"

    # Show secret age
    local creation=$(kubectl get secret "$secret" -n "$NAMESPACE" \
        -o jsonpath='{.metadata.creationTimestamp}')
    echo "  Created: $creation"

    # Show keys
    local keys=$(kubectl get secret "$secret" -n "$NAMESPACE" \
        -o jsonpath='{.data}' | grep -o '"[^"]*":' | tr -d '":')
    echo "  Keys:"
    echo "$keys" | while read -r key; do
        echo "    - $key"
    done

    # Check if any pods are using this secret
    print_info "Pods using this secret:"
    kubectl get pods -n "$NAMESPACE" -o json | \
        jq -r '.items[] | select(
            (.spec.volumes[]?.secret.secretName == "'$secret'") or
            (.spec.containers[].envFrom[]?.secretRef.name == "'$secret'")
        ) | "    - " + .metadata.name' 2>/dev/null || echo "    (none found)"

    echo ""
}

# Interactive menu
show_menu() {
    print_header "Secret Rotation Patterns"

    echo "Choose a rotation pattern to demonstrate:"
    echo ""
    echo "1) Zero-Downtime Rotation (Dual Secrets)"
    echo "2) Rolling Restart Rotation"
    echo "3) Immutable Secret Rotation"
    echo "4) External Secrets Operator (Simulation)"
    echo "5) Blue-Green Deployment Rotation"
    echo "6) Show Best Practices"
    echo "7) Verify Current Secrets"
    echo "8) Backup All Secrets"
    echo "9) Exit"
    echo ""
    read -p "Enter your choice (1-9): " choice

    case $choice in
        1) rotate_with_dual_secrets ;;
        2) rotate_with_rolling_restart ;;
        3) rotate_with_immutable_secrets ;;
        4) rotate_with_external_secrets ;;
        5) rotate_with_blue_green ;;
        6) show_best_practices ;;
        7)
            verify_rotation "$SECRET_NAME"
            verify_rotation "external-services"
            ;;
        8)
            print_header "Backing Up All Secrets"
            kubectl get secrets -n "$NAMESPACE" -o name | while read -r secret; do
                secret_name=$(echo "$secret" | cut -d'/' -f2)
                if [[ ! $secret_name =~ ^default-token ]]; then
                    backup_secret "$secret_name"
                fi
            done
            ;;
        9)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            return 1
            ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Main execution
main() {
    print_header "Kubernetes Secret Rotation Tool"

    print_info "Namespace: $NAMESPACE"
    echo ""

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_error "Namespace '$NAMESPACE' not found"
        print_info "Please run deploy-all.sh first"
        exit 1
    fi

    # Show menu
    show_menu
}

# Run main function
main "$@"
