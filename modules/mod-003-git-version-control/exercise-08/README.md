# Exercise 08: Git LFS for ML Projects - Implementation Guide

## Status: Implementation Guide In Progress

This exercise focuses on Git LFS (Large File Storage) specifically for ML model files and datasets.

## Topics Covered

- Git LFS installation and configuration
- Tracking model weights and checkpoints
- LFS storage backends (local, S3, Azure, GCP)
- Migrating existing large files to LFS
- LFS with CI/CD pipelines
- Managing LFS storage quotas
- LFS vs DVC comparison
- Best practices for ML artifacts

## Implementation Guide

A comprehensive step-by-step implementation guide for this exercise will be added in the next remediation phase.

For now, refer to the learning repository exercise file:
`/learning/ai-infra-junior-engineer-learning/lessons/mod-003-git-version-control/exercises/exercise-08-git-lfs-ml-projects.md`

## Quick Reference

Key commands covered:
```bash
git lfs install              # Setup LFS
git lfs track "*.pth"        # Track pattern
git lfs ls-files             # List LFS files
git lfs pull                 # Download LFS files
git lfs prune                # Clean old versions
git lfs migrate import       # Convert existing files
```

## Note

While this exercise provides dedicated Git LFS coverage, the core LFS concepts are also covered in Exercise 06 (ML Workflows with DVC and LFS). See exercise-06/IMPLEMENTATION_GUIDE.md for detailed LFS examples integrated with DVC workflows.

