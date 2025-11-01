#!/bin/bash
# Check Git LFS status and storage

echo "=== Git LFS Files ==="
git lfs ls-files

echo ""
echo "=== LFS Files with Sizes ==="
git lfs ls-files --size | head -20

echo ""
echo "=== LFS Storage Usage ==="
du -sh .git/lfs 2>/dev/null || echo "No LFS cache yet"

echo ""
echo "=== LFS Environment ==="
git lfs env | grep -E "Endpoint|LocalWorkingDir"
