#!/bin/bash

#######################################
# Backup Manager
#
# Automated backup and restore for Docker volumes,
# databases, and application state.
#
# Features:
# - Volume backups
# - Database backups (PostgreSQL, MySQL, MongoDB)
# - Incremental backups
# - S3 upload
# - Automated retention
# - Restore functionality
#######################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
S3_BUCKET="${S3_BUCKET:-}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $*"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $*"
}

#######################################
# Volume Backup Functions
#######################################

backup_volume() {
    local volume_name=$1
    local backup_path="${BACKUP_DIR}/volumes/${TIMESTAMP}"

    mkdir -p "$backup_path"

    log "Backing up volume: $volume_name"

    if ! docker volume inspect "$volume_name" &>/dev/null; then
        log_error "Volume $volume_name does not exist"
        return 1
    fi

    local backup_file="${backup_path}/${volume_name}.tar.gz"

    docker run --rm \
        -v "${volume_name}:/data:ro" \
        -v "${backup_path}:/backup" \
        alpine \
        tar czf "/backup/${volume_name}.tar.gz" -C /data .

    if [ -f "$backup_file" ]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "Volume $volume_name backed up successfully ($size)"
        echo "$backup_file"
        return 0
    else
        log_error "Failed to create backup for $volume_name"
        return 1
    fi
}

backup_all_volumes() {
    log "${BOLD}Backing up all Docker volumes${NC}"

    local backup_path="${BACKUP_DIR}/volumes/${TIMESTAMP}"
    mkdir -p "$backup_path"

    local volumes=$(docker volume ls -q)
    local count=0
    local failed=0

    for volume in $volumes; do
        if backup_volume "$volume"; then
            ((count++))
        else
            ((failed++))
        fi
    done

    log_success "Backed up $count volumes ($failed failed)"

    # Create manifest
    cat > "${backup_path}/manifest.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "type": "volumes",
  "volumes_count": $count,
  "failed_count": $failed
}
EOF

    echo "$backup_path"
}

restore_volume() {
    local volume_name=$1
    local backup_file=$2

    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    log "Restoring volume: $volume_name from $backup_file"

    # Create volume if doesn't exist
    if ! docker volume inspect "$volume_name" &>/dev/null; then
        log "Creating volume: $volume_name"
        docker volume create "$volume_name"
    fi

    # Stop containers using this volume
    local containers=$(docker ps -q --filter volume="$volume_name")
    if [ -n "$containers" ]; then
        log_warning "Stopping containers using volume $volume_name"
        for container in $containers; do
            docker stop "$container"
        done
    fi

    # Restore data
    docker run --rm \
        -v "${volume_name}:/data" \
        -v "$(dirname "$backup_file"):/backup:ro" \
        alpine \
        sh -c "rm -rf /data/* && tar xzf /backup/$(basename "$backup_file") -C /data"

    log_success "Volume $volume_name restored successfully"

    # Restart containers
    if [ -n "$containers" ]; then
        log "Restarting containers..."
        for container in $containers; do
            docker start "$container"
        done
    fi
}

#######################################
# Database Backup Functions
#######################################

backup_postgres() {
    local container_name=$1
    local database=${2:-all}
    local backup_path="${BACKUP_DIR}/postgres/${TIMESTAMP}"

    mkdir -p "$backup_path"

    log "Backing up PostgreSQL database from container: $container_name"

    if ! docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        log_error "Container $container_name is not running"
        return 1
    fi

    local backup_file="${backup_path}/${database}_${container_name}.sql.gz"

    if [ "$database" = "all" ]; then
        docker exec "$container_name" pg_dumpall -U postgres | gzip > "$backup_file"
    else
        docker exec "$container_name" pg_dump -U postgres "$database" | gzip > "$backup_file"
    fi

    if [ -f "$backup_file" ]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "PostgreSQL backup completed ($size)"
        echo "$backup_file"
        return 0
    else
        log_error "Failed to create PostgreSQL backup"
        return 1
    fi
}

restore_postgres() {
    local container_name=$1
    local backup_file=$2
    local database=${3:-postgres}

    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    log "Restoring PostgreSQL database to container: $container_name"

    if [[ "$backup_file" == *"all_"* ]]; then
        gunzip < "$backup_file" | docker exec -i "$container_name" psql -U postgres
    else
        # Drop and recreate database
        docker exec "$container_name" psql -U postgres -c "DROP DATABASE IF EXISTS $database;"
        docker exec "$container_name" psql -U postgres -c "CREATE DATABASE $database;"
        gunzip < "$backup_file" | docker exec -i "$container_name" psql -U postgres -d "$database"
    fi

    log_success "PostgreSQL database restored successfully"
}

backup_mysql() {
    local container_name=$1
    local database=${2:-all}
    local backup_path="${BACKUP_DIR}/mysql/${TIMESTAMP}"

    mkdir -p "$backup_path"

    log "Backing up MySQL database from container: $container_name"

    local backup_file="${backup_path}/${database}_${container_name}.sql.gz"

    if [ "$database" = "all" ]; then
        docker exec "$container_name" mysqldump -u root --all-databases | gzip > "$backup_file"
    else
        docker exec "$container_name" mysqldump -u root "$database" | gzip > "$backup_file"
    fi

    if [ -f "$backup_file" ]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "MySQL backup completed ($size)"
        echo "$backup_file"
        return 0
    else
        log_error "Failed to create MySQL backup"
        return 1
    fi
}

backup_mongodb() {
    local container_name=$1
    local database=${2:-all}
    local backup_path="${BACKUP_DIR}/mongodb/${TIMESTAMP}"

    mkdir -p "$backup_path"

    log "Backing up MongoDB database from container: $container_name"

    if [ "$database" = "all" ]; then
        docker exec "$container_name" mongodump --archive | gzip > "${backup_path}/mongodb_all.gz"
    else
        docker exec "$container_name" mongodump --db "$database" --archive | gzip > "${backup_path}/${database}.gz"
    fi

    log_success "MongoDB backup completed"
    echo "$backup_path"
}

#######################################
# S3 Upload Functions
#######################################

upload_to_s3() {
    local local_path=$1
    local s3_path="${S3_BUCKET}/$(basename "$local_path")"

    if [ -z "$S3_BUCKET" ]; then
        log_warning "S3_BUCKET not set, skipping upload"
        return 0
    fi

    log "Uploading to S3: $s3_path"

    if command -v aws &>/dev/null; then
        if [ -d "$local_path" ]; then
            aws s3 sync "$local_path" "s3://${s3_path}/" --storage-class STANDARD_IA
        else
            aws s3 cp "$local_path" "s3://${s3_path}" --storage-class STANDARD_IA
        fi

        log_success "Upload to S3 completed"
        return 0
    else
        log_error "AWS CLI not installed"
        return 1
    fi
}

download_from_s3() {
    local s3_path=$1
    local local_path=$2

    if [ -z "$S3_BUCKET" ]; then
        log_error "S3_BUCKET not set"
        return 1
    fi

    log "Downloading from S3: s3://${S3_BUCKET}/${s3_path}"

    mkdir -p "$local_path"

    if command -v aws &>/dev/null; then
        aws s3 cp "s3://${S3_BUCKET}/${s3_path}" "$local_path" --recursive
        log_success "Download from S3 completed"
        return 0
    else
        log_error "AWS CLI not installed"
        return 1
    fi
}

#######################################
# Cleanup Functions
#######################################

cleanup_old_backups() {
    local retention_days=${1:-$RETENTION_DAYS}

    log "Cleaning up backups older than $retention_days days"

    local deleted=0

    # Local backups
    if [ -d "$BACKUP_DIR" ]; then
        while IFS= read -r -d '' backup; do
            rm -rf "$backup"
            ((deleted++))
            log "Deleted: $backup"
        done < <(find "$BACKUP_DIR" -maxdepth 2 -type d -mtime "+$retention_days" -print0)
    fi

    log_success "Deleted $deleted old backups"

    # S3 backups
    if [ -n "$S3_BUCKET" ] && command -v aws &>/dev/null; then
        log "Cleaning up S3 backups older than $retention_days days"

        local cutoff_date=$(date -d "$retention_days days ago" +%Y-%m-%d)

        aws s3 ls "s3://${S3_BUCKET}/" --recursive | while read -r line; do
            local file_date=$(echo "$line" | awk '{print $1}')
            local file_path=$(echo "$line" | awk '{print $4}')

            if [[ "$file_date" < "$cutoff_date" ]]; then
                aws s3 rm "s3://${S3_BUCKET}/${file_path}"
                log "Deleted from S3: $file_path"
            fi
        done
    fi
}

#######################################
# Full Backup Function
#######################################

backup_full() {
    local backup_id="full_${TIMESTAMP}"
    local backup_path="${BACKUP_DIR}/${backup_id}"

    mkdir -p "$backup_path"

    log "${BOLD}Starting full backup: $backup_id${NC}"

    # Backup volumes
    log "Backing up volumes..."
    backup_all_volumes

    # Backup databases (auto-discover)
    log "Discovering databases..."

    # PostgreSQL
    local postgres_containers=$(docker ps --filter "ancestor=postgres" --format "{{.Names}}")
    for container in $postgres_containers; do
        backup_postgres "$container" "all"
    done

    # MySQL
    local mysql_containers=$(docker ps --filter "ancestor=mysql" --format "{{.Names}}")
    for container in $mysql_containers; do
        backup_mysql "$container" "all"
    done

    # MongoDB
    local mongo_containers=$(docker ps --filter "ancestor=mongo" --format "{{.Names}}")
    for container in $mongo_containers; do
        backup_mongodb "$container" "all"
    done

    # Create backup manifest
    cat > "${backup_path}/manifest.json" <<EOF
{
  "backup_id": "$backup_id",
  "timestamp": "$(date -Iseconds)",
  "type": "full",
  "hostname": "$(hostname)",
  "docker_version": "$(docker --version)"
}
EOF

    # Calculate total size
    local total_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    log_success "Full backup completed: $backup_id (Total size: $total_size)"

    # Upload to S3
    if [ -n "$S3_BUCKET" ]; then
        upload_to_s3 "$BACKUP_DIR"
    fi

    # Cleanup old backups
    cleanup_old_backups

    echo "$backup_id"
}

#######################################
# List Backups
#######################################

list_backups() {
    echo -e "\n${BOLD}Available Backups${NC}\n"

    if [ ! -d "$BACKUP_DIR" ]; then
        log_warning "No backups directory found"
        return 0
    fi

    find "$BACKUP_DIR" -maxdepth 2 -name "manifest.json" | sort -r | while read -r manifest; do
        local backup_dir=$(dirname "$manifest")
        local backup_id=$(basename "$backup_dir")
        local timestamp=$(jq -r '.timestamp' "$manifest" 2>/dev/null || echo "unknown")
        local type=$(jq -r '.type' "$manifest" 2>/dev/null || echo "unknown")
        local size=$(du -sh "$backup_dir" | cut -f1)

        echo -e "${CYAN}$backup_id${NC}"
        echo "  Type: $type"
        echo "  Time: $timestamp"
        echo "  Size: $size"
        echo "  Path: $backup_dir"
        echo
    done
}

#######################################
# Usage
#######################################

usage() {
    cat <<EOF
${BOLD}Backup Manager${NC}

Usage: $0 <command> [options]

Commands:
  backup-volume <volume_name>            Backup specific volume
  backup-all-volumes                     Backup all volumes
  backup-postgres <container> [db]       Backup PostgreSQL database
  backup-mysql <container> [db]          Backup MySQL database
  backup-mongodb <container> [db]        Backup MongoDB database
  backup-full                            Full backup (all volumes and databases)

  restore-volume <volume> <backup_file>  Restore volume from backup
  restore-postgres <container> <file>    Restore PostgreSQL database

  upload <path>                          Upload backup to S3
  download <s3_path> <local_path>        Download backup from S3

  list                                   List available backups
  cleanup [days]                         Cleanup backups older than N days

Environment Variables:
  BACKUP_DIR        Backup directory (default: /backups)
  S3_BUCKET         S3 bucket for remote backups
  RETENTION_DAYS    Retention period in days (default: 30)

Examples:
  # Full backup
  $0 backup-full

  # Backup specific volume
  $0 backup-volume ml-data

  # Backup PostgreSQL
  $0 backup-postgres ml-postgres mldb

  # Restore volume
  $0 restore-volume ml-data /backups/volumes/20241024_120000/ml-data.tar.gz

  # List backups
  $0 list

  # Cleanup old backups
  $0 cleanup 7

EOF
}

#######################################
# Main
#######################################

main() {
    if [ $# -eq 0 ]; then
        usage
        exit 1
    fi

    local command=$1
    shift

    case "$command" in
        backup-volume)
            [ $# -lt 1 ] && { log_error "Missing volume name"; exit 1; }
            backup_volume "$1"
            ;;

        backup-all-volumes)
            backup_all_volumes
            ;;

        backup-postgres)
            [ $# -lt 1 ] && { log_error "Missing container name"; exit 1; }
            backup_postgres "$1" "${2:-all}"
            ;;

        backup-mysql)
            [ $# -lt 1 ] && { log_error "Missing container name"; exit 1; }
            backup_mysql "$1" "${2:-all}"
            ;;

        backup-mongodb)
            [ $# -lt 1 ] && { log_error "Missing container name"; exit 1; }
            backup_mongodb "$1" "${2:-all}"
            ;;

        backup-full)
            backup_full
            ;;

        restore-volume)
            [ $# -lt 2 ] && { log_error "Missing volume name or backup file"; exit 1; }
            restore_volume "$1" "$2"
            ;;

        restore-postgres)
            [ $# -lt 2 ] && { log_error "Missing container name or backup file"; exit 1; }
            restore_postgres "$1" "$2" "${3:-postgres}"
            ;;

        upload)
            [ $# -lt 1 ] && { log_error "Missing path"; exit 1; }
            upload_to_s3 "$1"
            ;;

        download)
            [ $# -lt 2 ] && { log_error "Missing S3 path or local path"; exit 1; }
            download_from_s3 "$1" "$2"
            ;;

        list)
            list_backups
            ;;

        cleanup)
            cleanup_old_backups "${1:-$RETENTION_DAYS}"
            ;;

        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
