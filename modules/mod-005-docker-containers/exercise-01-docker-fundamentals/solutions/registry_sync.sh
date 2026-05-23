#!/bin/bash
#
# registry_sync.sh - Docker registry synchronization tool
#
# Description:
#   Synchronize Docker images between registries, backup images,
#   and manage multi-registry deployments for ML infrastructure.
#
# Usage:
#   ./registry_sync.sh [OPTIONS]
#
# Options:
#   --source URL          Source registry URL
#   --dest URL            Destination registry URL
#   --images FILE         File with list of images to sync
#   --tag-filter PATTERN  Filter images by tag pattern
#   --parallel N          Number of parallel syncs (default: 3)
#   --dry-run             Show what would be synced
#   --verify              Verify sync after completion
#   -v, --verbose         Verbose output
#   -h, --help            Display this help
#

set -euo pipefail

# Configuration
SOURCE_REGISTRY=""
DEST_REGISTRY=""
IMAGES_FILE=""
TAG_FILTER="*"
PARALLEL=3
DRY_RUN=false
VERIFY=false
VERBOSE=false

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly RESET='\033[0m'
readonly BOLD='\033[1m'

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${RESET} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $*"
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${CYAN}[DEBUG]${RESET} $*"
    fi
}

# Sync single image
sync_image() {
    local source_image="$1"
    local dest_image="$2"

    log_info "Syncing: $source_image -> $dest_image"

    if [[ "$DRY_RUN" == true ]]; then
        log_warning "[DRY RUN] Would sync: $source_image -> $dest_image"
        return 0
    fi

    # Pull from source
    log_verbose "Pulling from source: $source_image"
    if ! docker pull "$source_image" 2>&1 | grep -v "Digest:"; then
        log_error "Failed to pull: $source_image"
        return 1
    fi

    # Tag for destination
    log_verbose "Tagging for destination: $dest_image"
    if ! docker tag "$source_image" "$dest_image"; then
        log_error "Failed to tag: $dest_image"
        return 1
    fi

    # Push to destination
    log_verbose "Pushing to destination: $dest_image"
    if ! docker push "$dest_image" 2>&1 | grep -v "Digest:"; then
        log_error "Failed to push: $dest_image"
        # Cleanup
        docker rmi "$dest_image" 2>/dev/null || true
        return 1
    fi

    # Cleanup local images
    docker rmi "$source_image" "$dest_image" 2>/dev/null || true

    log_success "Synced: $source_image -> $dest_image"
    return 0
}

# Verify image sync
verify_image() {
    local source_image="$1"
    local dest_image="$2"

    log_verbose "Verifying: $dest_image"

    # Get source digest
    local source_digest
    source_digest=$(docker pull "$source_image" 2>&1 | grep "Digest:" | awk '{print $2}')

    if [[ -z "$source_digest" ]]; then
        log_warning "Could not get source digest for $source_image"
        return 1
    fi

    # Get destination digest
    local dest_digest
    dest_digest=$(docker pull "$dest_image" 2>&1 | grep "Digest:" | awk '{print $2}')

    if [[ -z "$dest_digest" ]]; then
        log_error "Could not get destination digest for $dest_image"
        return 1
    fi

    # Compare digests
    if [[ "$source_digest" == "$dest_digest" ]]; then
        log_success "Verified: $dest_image (digest: $dest_digest)"
        return 0
    else
        log_error "Digest mismatch for $dest_image"
        log_error "  Source: $source_digest"
        log_error "  Dest:   $dest_digest"
        return 1
    fi
}

# Read images from file
read_images_file() {
    local file="$1"

    if [[ ! -f "$file" ]]; then
        log_error "Images file not found: $file"
        exit 1
    fi

    # Read and filter images
    local images=()
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue

        # Apply tag filter
        if [[ "$line" == $TAG_FILTER ]]; then
            images+=("$line")
        fi
    done < "$file"

    echo "${images[@]}"
}

# Sync multiple images in parallel
sync_images_parallel() {
    local -a images=("$@")
    local total=${#images[@]}
    local success=0
    local failed=0

    log_info "Syncing $total image(s) with parallelism $PARALLEL"

    # Temporary files for tracking
    local tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" EXIT

    local job_count=0
    local pids=()

    for image in "${images[@]}"; do
        # Wait if at max parallel jobs
        while [[ ${#pids[@]} -ge $PARALLEL ]]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}"
                    unset 'pids[i]'
                fi
            done
            pids=("${pids[@]}")  # Reindex array
            sleep 0.1
        done

        # Build source and destination image names
        local source_image="${SOURCE_REGISTRY}/${image}"
        local dest_image="${DEST_REGISTRY}/${image}"

        # Start sync in background
        (
            if sync_image "$source_image" "$dest_image"; then
                echo "success" > "$tmp_dir/result_$$"

                if [[ "$VERIFY" == true ]]; then
                    verify_image "$source_image" "$dest_image"
                fi
            else
                echo "failed" > "$tmp_dir/result_$$"
            fi
        ) &

        pids+=($!)
        ((job_count++))
    done

    # Wait for all jobs
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    # Count results
    success=$(ls "$tmp_dir"/result_* 2>/dev/null | wc -l || echo 0)
    failed=$((total - success))

    # Print summary
    echo ""
    log_info "Sync Summary:"
    log_info "  Total: $total"
    log_success "  Success: $success"
    if [[ $failed -gt 0 ]]; then
        log_error "  Failed: $failed"
    fi
}

# Main sync operation
main_sync() {
    echo -e "${BOLD}${CYAN}Docker Registry Sync${RESET}"
    echo "=========================================="
    echo "Source:      $SOURCE_REGISTRY"
    echo "Destination: $DEST_REGISTRY"
    echo "Parallel:    $PARALLEL"
    echo "Verify:      $VERIFY"
    echo "=========================================="
    echo ""

    # Login to registries
    log_info "Checking registry access..."

    # Test source registry
    if ! docker pull "${SOURCE_REGISTRY}/hello-world:latest" &>/dev/null; then
        log_warning "Could not access source registry, may need authentication"
    else
        docker rmi "${SOURCE_REGISTRY}/hello-world:latest" &>/dev/null || true
    fi

    # Test destination registry
    if ! docker pull "${DEST_REGISTRY}/hello-world:latest" &>/dev/null; then
        log_warning "Could not access destination registry, may need authentication"
    else
        docker rmi "${DEST_REGISTRY}/hello-world:latest" &>/dev/null || true
    fi

    # Read images to sync
    local images
    if [[ -n "$IMAGES_FILE" ]]; then
        images=($(read_images_file "$IMAGES_FILE"))
    else
        log_error "No images file specified"
        exit 1
    fi

    if [[ ${#images[@]} -eq 0 ]]; then
        log_error "No images to sync"
        exit 1
    fi

    # Perform sync
    sync_images_parallel "${images[@]}"

    log_success "Sync operation completed!"
}

# Usage
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Synchronize Docker images between registries.

OPTIONS:
    --source URL          Source registry URL
    --dest URL            Destination registry URL
    --images FILE         File with list of images to sync
    --tag-filter PATTERN  Filter images by tag pattern (default: *)
    --parallel N          Number of parallel syncs (default: 3)
    --dry-run             Show what would be synced
    --verify              Verify sync after completion
    -v, --verbose         Verbose output
    -h, --help            Display this help

EXAMPLES:
    # Sync images from file
    $0 --source registry.source.com --dest registry.dest.com --images images.txt

    # Dry run with verification
    $0 --source source.io --dest dest.io --images images.txt --dry-run --verify

    # Parallel sync with filter
    $0 --source hub.docker.com --dest private.registry.com \\
       --images ml-images.txt --tag-filter "*:v1.*" --parallel 5

IMAGES FILE FORMAT:
    # One image per line (without registry prefix)
    myapp/model:v1.0
    myapp/api:latest
    pytorch/pytorch:2.0-cuda11.8

EOF
}

# Argument parsing
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --source)
                SOURCE_REGISTRY="$2"
                shift 2
                ;;
            --dest)
                DEST_REGISTRY="$2"
                shift 2
                ;;
            --images)
                IMAGES_FILE="$2"
                shift 2
                ;;
            --tag-filter)
                TAG_FILTER="$2"
                shift 2
                ;;
            --parallel)
                PARALLEL="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verify)
                VERIFY=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Main
main() {
    parse_arguments "$@"

    # Validate required parameters
    if [[ -z "$SOURCE_REGISTRY" ]]; then
        log_error "Source registry is required"
        usage
        exit 1
    fi

    if [[ -z "$DEST_REGISTRY" ]]; then
        log_error "Destination registry is required"
        usage
        exit 1
    fi

    if [[ -z "$IMAGES_FILE" ]]; then
        log_error "Images file is required"
        usage
        exit 1
    fi

    main_sync
}

main "$@"
