#!/usr/bin/env python3
"""
Basic list operations with ML data.

Demonstrates fundamental list operations using ML training data as examples.
"""

from typing import List


def main():
    """Demonstrate basic list operations for ML data processing."""
    print("=" * 60)
    print("List Operations for ML Data Processing")
    print("=" * 60)
    print()

    # Sample dataset: image file paths for training
    training_images = [
        "img_0001.jpg",
        "img_0002.jpg",
        "img_0003.jpg",
        "img_0004.jpg",
        "img_0005.jpg"
    ]

    # Print dataset information
    print(f"Total training images: {len(training_images)}")
    print(f"First image: {training_images[0]}")
    print(f"Last image: {training_images[-1]}")
    print()

    # Add new images
    training_images.append("img_0006.jpg")
    training_images.extend(["img_0007.jpg", "img_0008.jpg"])
    print(f"After adding: {len(training_images)} images")

    # Insert image at specific position
    training_images.insert(0, "img_0000.jpg")
    print(f"First image now: {training_images[0]}")
    print()

    # Remove images
    removed = training_images.pop()  # Remove last
    print(f"Removed: {removed}")

    training_images.remove("img_0000.jpg")  # Remove by value
    print(f"Final count: {len(training_images)}")
    print()

    # Check if image exists
    if "img_0003.jpg" in training_images:
        index = training_images.index("img_0003.jpg")
        print(f"Found img_0003.jpg at index {index}")
    print()

    # Slice operations (get batches)
    batch_size = 3
    batch_1 = training_images[0:batch_size]
    batch_2 = training_images[batch_size:batch_size*2]
    print(f"Batch size: {batch_size}")
    print(f"Batch 1: {batch_1}")
    print(f"Batch 2: {batch_2}")
    print()

    # Reverse and sort
    training_images_sorted = sorted(training_images)
    print(f"Sorted: {training_images_sorted}")

    training_images_reversed = list(reversed(training_images))
    print(f"Reversed: {training_images_reversed}")
    print()

    # Extended tasks
    print("=" * 60)
    print("Extended Tasks")
    print("=" * 60)
    print()

    # Task 1: Add 10 more images
    for i in range(9, 19):
        training_images.append(f"img_{i:04d}.jpg")
    print(f"After adding 10 more: {len(training_images)} images")

    # Task 2: Create batches of size 4
    batch_size = 4
    batches = [training_images[i:i+batch_size]
               for i in range(0, len(training_images), batch_size)]
    print(f"Created {len(batches)} batches of size {batch_size}")
    print(f"First batch: {batches[0]}")
    print(f"Last batch size: {len(batches[-1])}")
    print()

    # Task 3: Find all images with "000" in their name
    images_with_000 = [img for img in training_images if "000" in img]
    print(f"Images containing '000': {images_with_000}")
    print()

    # Additional useful operations
    print("=" * 60)
    print("Additional Operations")
    print("=" * 60)
    print()

    # Count occurrences
    count_jpg = sum(1 for img in training_images if img.endswith(".jpg"))
    print(f"Total .jpg files: {count_jpg}")

    # Filter by pattern
    images_0001_to_0005 = [img for img in training_images
                          if "0001" in img or "0002" in img or
                             "0003" in img or "0004" in img or "0005" in img]
    print(f"Images 0001-0005: {len(images_0001_to_0005)} images")

    # Get every nth image
    n = 3
    every_nth = training_images[::n]
    print(f"Every {n}rd image: {every_nth}")

    print()
    print("✓ List operations demonstration complete")


if __name__ == "__main__":
    main()
