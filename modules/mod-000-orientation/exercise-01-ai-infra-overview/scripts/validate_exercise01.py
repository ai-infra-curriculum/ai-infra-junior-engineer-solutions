#!/usr/bin/env python3
"""
Validation script for Exercise 01: AI Infrastructure Overview & Career Paths

This script checks if all required deliverables have been completed.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(message: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def check_file_exists(file_path: Path, min_size: int = 100) -> Tuple[bool, str]:
    """
    Check if a file exists and has minimum content.

    Args:
        file_path: Path to file
        min_size: Minimum file size in bytes

    Returns:
        Tuple of (success, message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path.name}"

    size = file_path.stat().st_size
    if size < min_size:
        return False, f"File too small ({size} bytes): {file_path.name}"

    return True, f"Found: {file_path.name} ({size} bytes)"


def check_skills_assessment(docs_dir: Path) -> bool:
    """Check if skills assessment is completed."""
    print(f"\n{Colors.BOLD}Checking Skills Assessment...{Colors.END}")

    # Check for completed assessment file
    completed_file = docs_dir / "skills-assessment-completed.md"
    template_file = docs_dir / "skills-assessment.md"

    if completed_file.exists():
        success, message = check_file_exists(completed_file, min_size=3000)
        if success:
            print_success(message)

            # Check if file has been filled out (look for non-empty ratings)
            content = completed_file.read_text()
            if "____" in content and content.count("____") > 50:
                print_warning("Skills assessment appears incomplete (many blank ratings)")
                print_warning("Make sure to fill out all rating sections")
                return False
            else:
                print_success("Skills assessment appears complete")
                return True
        else:
            print_error(message)
            return False
    else:
        print_error(f"Skills assessment not found: {completed_file.name}")
        print_warning(f"Copy {template_file.name} to {completed_file.name} and complete it")
        return False


def check_learning_plan(docs_dir: Path) -> bool:
    """Check if learning plan is created."""
    print(f"\n{Colors.BOLD}Checking Learning Plan...{Colors.END}")

    learning_plan = docs_dir / "learning-plan.md"
    template = docs_dir / "learning-plan-template.md"

    if learning_plan.exists():
        success, message = check_file_exists(learning_plan, min_size=3000)
        if success:
            print_success(message)

            # Check if plan has been personalized
            content = learning_plan.read_text()
            if "Name:" in content and "_______________________________" in content:
                print_warning("Learning plan appears to be template (not personalized)")
                print_warning("Fill in your name and customize the plan")
                return False
            else:
                print_success("Learning plan appears personalized")
                return True
        else:
            print_error(message)
            return False
    else:
        print_error(f"Learning plan not found: {learning_plan.name}")
        print_warning(f"Copy {template.name} to {learning_plan.name} and customize it")
        return False


def check_career_roadmap(docs_dir: Path) -> bool:
    """Check if career roadmap is defined."""
    print(f"\n{Colors.BOLD}Checking Career Roadmap...{Colors.END}")

    roadmap = docs_dir / "career-roadmap.md"
    template = docs_dir / "career-roadmap-template.md"

    if roadmap.exists():
        success, message = check_file_exists(roadmap, min_size=3000)
        if success:
            print_success(message)

            # Check if roadmap has been personalized
            content = roadmap.read_text()
            if "Name:" in content and "_______________________________" in content:
                print_warning("Career roadmap appears to be template (not personalized)")
                print_warning("Fill in your personal information and goals")
                return False
            else:
                print_success("Career roadmap appears personalized")
                return True
        else:
            print_error(message)
            return False
    else:
        print_error(f"Career roadmap not found: {roadmap.name}")
        print_warning(f"Copy {template.name} to {roadmap.name} and customize it")
        return False


def check_job_research(docs_dir: Path) -> bool:
    """Check if job market research is done."""
    print(f"\n{Colors.BOLD}Checking Job Market Research...{Colors.END}")

    research_file = docs_dir / "job-market-research.md"

    if research_file.exists():
        success, message = check_file_exists(research_file, min_size=1000)
        if success:
            print_success(message)

            # Check if research has content
            content = research_file.read_text()
            if "# Job Market Research" in content:
                print_success("Job market research found")
                return True
            else:
                print_warning("Job market research may need more detail")
                return False
        else:
            print_error(message)
            return False
    else:
        print_warning(f"Job market research not found: {research_file.name}")
        print_warning("This deliverable is optional but recommended")
        print_warning("Create a file documenting 10 job postings you researched")
        return False  # Optional, so don't fail validation


def run_validation():
    """Run all validation checks."""
    print_header("Exercise 01 Validation")

    # Get docs directory
    script_dir = Path(__file__).parent
    exercise_dir = script_dir.parent
    docs_dir = exercise_dir / "docs"

    if not docs_dir.exists():
        print_error(f"Docs directory not found: {docs_dir}")
        sys.exit(1)

    # Run checks
    checks = {
        "Skills Assessment": check_skills_assessment(docs_dir),
        "Learning Plan": check_learning_plan(docs_dir),
        "Career Roadmap": check_career_roadmap(docs_dir),
        "Job Market Research": check_job_research(docs_dir),  # Optional
    }

    # Summary
    print_header("Validation Summary")

    passed = sum(1 for result in checks.values() if result)
    total = len(checks)

    for check_name, result in checks.items():
        if result:
            print_success(f"{check_name}: Complete")
        else:
            if check_name == "Job Market Research":
                print_warning(f"{check_name}: Optional (not required)")
            else:
                print_error(f"{check_name}: Incomplete")

    # Final status
    required_checks = {k: v for k, v in checks.items() if k != "Job Market Research"}
    required_passed = sum(1 for result in required_checks.values() if result)
    required_total = len(required_checks)

    print(f"\n{Colors.BOLD}Score: {required_passed}/{required_total} required deliverables complete{Colors.END}")

    if required_passed == required_total:
        print_success("\n🎉 Exercise 01 Complete! Ready for Exercise 02.")
        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        print("1. Move to Exercise 02: Development Environment Setup")
        print("2. Start implementing your learning plan")
        print("3. Join online communities")
        print("4. Begin building your portfolio")
        return 0
    else:
        print_error(f"\n❌ Exercise incomplete. {required_total - required_passed} deliverable(s) remaining.")
        print(f"\n{Colors.BOLD}To Do:{Colors.END}")
        for check_name, result in required_checks.items():
            if not result:
                print(f"  - Complete {check_name}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = run_validation()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Validation interrupted.{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Validation error: {e}")
        sys.exit(1)
