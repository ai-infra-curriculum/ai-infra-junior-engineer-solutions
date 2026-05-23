#!/usr/bin/env python3
"""
image_cleaner.py - Automated Docker image cleanup tool

Description:
    Clean up unused Docker images, optimize disk space, and manage
    image lifecycle with intelligent cleanup strategies.

Usage:
    python image_cleaner.py [OPTIONS]

Options:
    --dangling          Remove dangling images only
    --unused            Remove unused images
    --old-days N        Remove images older than N days
    --size-threshold GB Remove images if total exceeds threshold
    --dry-run           Show what would be removed
    --keep-tagged       Keep all tagged images
    --verbose           Verbose output
    --help              Display help
"""

import docker
import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import defaultdict

# Colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class ImageCleaner:
    """Docker image cleanup automation"""

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        """Initialize image cleaner"""
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:
            print(f"{Colors.RED}Error connecting to Docker: {e}{Colors.RESET}")
            sys.exit(1)

        self.dry_run = dry_run
        self.verbose = verbose
        self.total_space_freed = 0

    def get_image_info(self, image) -> Dict:
        """Get detailed image information"""
        try:
            size_bytes = image.attrs.get('Size', 0)
            size_mb = size_bytes / (1024 * 1024)

            created_str = image.attrs.get('Created', '')
            created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))

            tags = image.tags if image.tags else ['<none>']

            return {
                'id': image.short_id,
                'tags': tags,
                'size_mb': size_mb,
                'size_bytes': size_bytes,
                'created': created,
                'age_days': (datetime.now(created.tzinfo) - created).days
            }
        except Exception as e:
            if self.verbose:
                print(f"{Colors.YELLOW}Warning: Error getting image info: {e}{Colors.RESET}")
            return None

    def list_dangling_images(self) -> List:
        """List dangling images (untagged)"""
        try:
            dangling = self.client.images.list(filters={'dangling': True})
            return dangling
        except Exception as e:
            print(f"{Colors.RED}Error listing dangling images: {e}{Colors.RESET}")
            return []

    def list_unused_images(self) -> List:
        """List unused images (not used by any container)"""
        try:
            # Get images used by containers
            containers = self.client.containers.list(all=True)
            used_images = set()
            for container in containers:
                used_images.add(container.image.id)

            # Find unused images
            all_images = self.client.images.list()
            unused = [img for img in all_images if img.id not in used_images]

            return unused
        except Exception as e:
            print(f"{Colors.RED}Error listing unused images: {e}{Colors.RESET}")
            return []

    def list_old_images(self, days: int) -> List:
        """List images older than specified days"""
        try:
            all_images = self.client.images.list()
            old_images = []

            cutoff_date = datetime.now() - timedelta(days=days)

            for image in all_images:
                info = self.get_image_info(image)
                if info and info['created'] < cutoff_date.replace(tzinfo=info['created'].tzinfo):
                    old_images.append(image)

            return old_images
        except Exception as e:
            print(f"{Colors.RED}Error listing old images: {e}{Colors.RESET}")
            return []

    def remove_images(self, images: List, reason: str) -> Tuple[int, float]:
        """Remove images and return count and space freed"""
        if not images:
            return 0, 0

        print(f"\n{Colors.BOLD}Found {len(images)} image(s) to remove ({reason}):{Colors.RESET}")

        removed_count = 0
        space_freed = 0

        for image in images:
            info = self.get_image_info(image)
            if not info:
                continue

            tags_str = ', '.join(info['tags'])
            size_str = f"{info['size_mb']:.1f} MB"

            if self.dry_run:
                print(f"  {Colors.YELLOW}[DRY RUN]{Colors.RESET} Would remove: {tags_str} ({info['id']}) - {size_str}")
            else:
                try:
                    self.client.images.remove(image.id, force=True)
                    print(f"  {Colors.GREEN}✓{Colors.RESET} Removed: {tags_str} ({info['id']}) - {size_str}")
                    removed_count += 1
                    space_freed += info['size_bytes']
                except Exception as e:
                    print(f"  {Colors.RED}✗{Colors.RESET} Failed to remove {info['id']}: {e}")

        return removed_count, space_freed

    def clean_dangling(self) -> None:
        """Clean dangling images"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Cleaning Dangling Images{Colors.RESET}")
        print("=" * 80)

        images = self.list_dangling_images()
        count, space = self.remove_images(images, "dangling/untagged")

        if not self.dry_run and count > 0:
            self.total_space_freed += space
            print(f"\n{Colors.GREEN}Removed {count} dangling image(s), freed {space / (1024 * 1024):.1f} MB{Colors.RESET}")
        elif images and self.dry_run:
            total_size = sum(self.get_image_info(img)['size_bytes'] for img in images if self.get_image_info(img))
            print(f"\n{Colors.YELLOW}Would free {total_size / (1024 * 1024):.1f} MB{Colors.RESET}")

    def clean_unused(self, keep_tagged: bool = True) -> None:
        """Clean unused images"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Cleaning Unused Images{Colors.RESET}")
        print("=" * 80)

        images = self.list_unused_images()

        # Filter out tagged images if keep_tagged is True
        if keep_tagged:
            images = [img for img in images if not img.tags]

        count, space = self.remove_images(images, "unused")

        if not self.dry_run and count > 0:
            self.total_space_freed += space
            print(f"\n{Colors.GREEN}Removed {count} unused image(s), freed {space / (1024 * 1024):.1f} MB{Colors.RESET}")
        elif images and self.dry_run:
            total_size = sum(self.get_image_info(img)['size_bytes'] for img in images if self.get_image_info(img))
            print(f"\n{Colors.YELLOW}Would free {total_size / (1024 * 1024):.1f} MB{Colors.RESET}")

    def clean_old(self, days: int, keep_tagged: bool = True) -> None:
        """Clean old images"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Cleaning Images Older Than {days} Days{Colors.RESET}")
        print("=" * 80)

        images = self.list_old_images(days)

        # Filter out tagged images if keep_tagged is True
        if keep_tagged:
            images = [img for img in images if not img.tags]

        count, space = self.remove_images(images, f"older than {days} days")

        if not self.dry_run and count > 0:
            self.total_space_freed += space
            print(f"\n{Colors.GREEN}Removed {count} old image(s), freed {space / (1024 * 1024):.1f} MB{Colors.RESET}")
        elif images and self.dry_run:
            total_size = sum(self.get_image_info(img)['size_bytes'] for img in images if self.get_image_info(img))
            print(f"\n{Colors.YELLOW}Would free {total_size / (1024 * 1024):.1f} MB{Colors.RESET}")

    def clean_by_size_threshold(self, threshold_gb: float, keep_tagged: bool = True) -> None:
        """Clean images if total size exceeds threshold"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Cleaning to Meet Size Threshold{Colors.RESET}")
        print("=" * 80)

        # Get all images sorted by age (oldest first)
        all_images = self.client.images.list()
        image_infos = []

        for image in all_images:
            info = self.get_image_info(image)
            if info:
                image_infos.append((image, info))

        # Sort by age (oldest first)
        image_infos.sort(key=lambda x: x[1]['created'])

        # Calculate current total size
        total_size_gb = sum(info['size_bytes'] for _, info in image_infos) / (1024 ** 3)

        print(f"Current total image size: {total_size_gb:.2f} GB")
        print(f"Threshold: {threshold_gb:.2f} GB")

        if total_size_gb <= threshold_gb:
            print(f"\n{Colors.GREEN}Size is within threshold, no cleanup needed{Colors.RESET}")
            return

        print(f"\n{Colors.YELLOW}Exceeds threshold by {total_size_gb - threshold_gb:.2f} GB{Colors.RESET}")

        # Remove oldest images until under threshold
        images_to_remove = []
        current_size = total_size_gb

        for image, info in image_infos:
            if current_size <= threshold_gb:
                break

            # Skip tagged images if keep_tagged
            if keep_tagged and info['tags'] != ['<none>']:
                continue

            images_to_remove.append(image)
            current_size -= info['size_bytes'] / (1024 ** 3)

        count, space = self.remove_images(images_to_remove, "to meet size threshold")

        if not self.dry_run and count > 0:
            self.total_space_freed += space
            new_size = total_size_gb - (space / (1024 ** 3))
            print(f"\n{Colors.GREEN}Removed {count} image(s), new size: {new_size:.2f} GB{Colors.RESET}")

    def show_statistics(self) -> None:
        """Show image statistics"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Docker Image Statistics{Colors.RESET}")
        print("=" * 80)

        try:
            all_images = self.client.images.list()
            dangling = self.list_dangling_images()
            unused = self.list_unused_images()

            # Calculate total size
            total_size = sum(img.attrs.get('Size', 0) for img in all_images)
            dangling_size = sum(img.attrs.get('Size', 0) for img in dangling)
            unused_size = sum(img.attrs.get('Size', 0) for img in unused)

            print(f"\nTotal Images: {len(all_images)}")
            print(f"  Total Size: {total_size / (1024 ** 3):.2f} GB")
            print(f"\nDangling Images: {len(dangling)}")
            print(f"  Size: {dangling_size / (1024 ** 3):.2f} GB")
            print(f"\nUnused Images: {len(unused)}")
            print(f"  Size: {unused_size / (1024 ** 3):.2f} GB")

            # Age distribution
            age_buckets = defaultdict(int)
            for image in all_images:
                info = self.get_image_info(image)
                if info:
                    age = info['age_days']
                    if age < 7:
                        age_buckets['< 1 week'] += 1
                    elif age < 30:
                        age_buckets['1-4 weeks'] += 1
                    elif age < 90:
                        age_buckets['1-3 months'] += 1
                    else:
                        age_buckets['> 3 months'] += 1

            print(f"\nAge Distribution:")
            for bucket, count in sorted(age_buckets.items()):
                print(f"  {bucket}: {count}")

        except Exception as e:
            print(f"{Colors.RED}Error showing statistics: {e}{Colors.RESET}")

    def print_summary(self) -> None:
        """Print cleanup summary"""
        if self.total_space_freed > 0:
            print(f"\n{Colors.BOLD}{Colors.GREEN}Cleanup Summary:{Colors.RESET}")
            print(f"  Total space freed: {self.total_space_freed / (1024 ** 3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description='Docker image cleanup tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dangling', action='store_true',
                       help='Remove dangling images only')
    parser.add_argument('--unused', action='store_true',
                       help='Remove unused images')
    parser.add_argument('--old-days', type=int, default=None,
                       help='Remove images older than N days')
    parser.add_argument('--size-threshold', type=float, default=None,
                       help='Remove images if total exceeds threshold (GB)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be removed')
    parser.add_argument('--keep-tagged', action='store_true', default=True,
                       help='Keep all tagged images (default: True)')
    parser.add_argument('--stats', action='store_true',
                       help='Show image statistics')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Create cleaner
    cleaner = ImageCleaner(dry_run=args.dry_run, verbose=args.verbose)

    if args.dry_run:
        print(f"{Colors.YELLOW}{Colors.BOLD}DRY RUN MODE - No images will be removed{Colors.RESET}\n")

    # Show statistics if requested
    if args.stats:
        cleaner.show_statistics()
        return

    # Perform cleanup operations
    if args.dangling:
        cleaner.clean_dangling()

    if args.unused:
        cleaner.clean_unused(keep_tagged=args.keep_tagged)

    if args.old_days:
        cleaner.clean_old(args.old_days, keep_tagged=args.keep_tagged)

    if args.size_threshold:
        cleaner.clean_by_size_threshold(args.size_threshold, keep_tagged=args.keep_tagged)

    # If no specific cleanup requested, show statistics
    if not any([args.dangling, args.unused, args.old_days, args.size_threshold]):
        cleaner.show_statistics()
        print(f"\n{Colors.YELLOW}No cleanup operations specified. Use --help for options.{Colors.RESET}")
    else:
        cleaner.print_summary()


if __name__ == '__main__':
    main()
