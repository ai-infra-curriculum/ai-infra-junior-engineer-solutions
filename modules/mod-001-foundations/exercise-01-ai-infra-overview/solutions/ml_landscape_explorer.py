#!/usr/bin/env python3
"""
AI/ML Landscape Explorer - Interactive tool for understanding ML frameworks and tools

This tool helps aspiring AI infrastructure engineers explore:
- ML frameworks (TensorFlow, PyTorch, scikit-learn, etc.)
- Infrastructure tools (Docker, Kubernetes, MLflow, etc.)
- Career paths and required skills
- Industry trends and job market

Usage:
    python ml_landscape_explorer.py                 # Interactive mode
    python ml_landscape_explorer.py --frameworks    # List frameworks
    python ml_landscape_explorer.py --compare "PyTorch" "TensorFlow"
    python ml_landscape_explorer.py --quiz          # Test your knowledge

Author: AI Infrastructure Curriculum Team
License: MIT
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import random


class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")


def print_section(text: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'-'*len(text)}{Colors.END}")


def load_data(filename: str) -> Optional[Dict]:
    """Load JSON data file."""
    data_path = Path(__file__).parent / 'data' / filename
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Colors.RED}Error: Data file '{filename}' not found{Colors.END}")
        return None
    except json.JSONDecodeError:
        print(f"{Colors.RED}Error: Invalid JSON in '{filename}'{Colors.END}")
        return None


class MLLandscapeExplorer:
    """Main explorer class."""

    def __init__(self):
        """Initialize explorer with data."""
        self.frameworks_data = load_data('frameworks.json')
        if not self.frameworks_data:
            print(f"{Colors.RED}Failed to load frameworks data{Colors.END}")
            sys.exit(1)

        self.frameworks = self.frameworks_data.get('ml_frameworks', [])

    def list_frameworks(self, filter_type: Optional[str] = None):
        """List all ML frameworks with basic info."""
        print_header("ML Frameworks Overview")

        frameworks = self.frameworks
        if filter_type:
            frameworks = [f for f in frameworks if f.get('type') == filter_type]

        if not frameworks:
            print(f"{Colors.YELLOW}No frameworks found for type: {filter_type}{Colors.END}")
            return

        # Group by type
        by_type = {}
        for fw in frameworks:
            fw_type = fw.get('type', 'Other')
            if fw_type not in by_type:
                by_type[fw_type] = []
            by_type[fw_type].append(fw)

        for fw_type, fws in sorted(by_type.items()):
            print_section(f"{fw_type} Frameworks")
            for fw in fws:
                stars = fw.get('github_stars', 0)
                stars_str = f"{stars//1000}k" if stars >= 1000 else str(stars)
                demand = fw.get('job_demand', 'Unknown')
                market = fw.get('market_share', 'N/A')

                print(f"\n{Colors.BOLD}{Colors.GREEN}• {fw['name']}{Colors.END}")
                print(f"  Developer: {fw.get('developer', 'N/A')}")
                print(f"  Language: {fw.get('language', 'N/A')}")
                print(f"  GitHub Stars: ⭐ {stars_str}")
                print(f"  Job Demand: {self._color_demand(demand)}")
                print(f"  Market Share: {market}")
                print(f"  Learning Curve: {fw.get('learning_curve', 'N/A')}")

    def _color_demand(self, demand: str) -> str:
        """Color code job demand."""
        colors = {
            'Very High': Colors.GREEN,
            'High': Colors.GREEN,
            'Medium-High': Colors.YELLOW,
            'Medium': Colors.YELLOW,
            'Low-Medium': Colors.RED,
            'Low': Colors.RED
        }
        color = colors.get(demand, '')
        return f"{color}{demand}{Colors.END}"

    def show_framework_details(self, name: str):
        """Show detailed information about a framework."""
        fw = self._find_framework(name)
        if not fw:
            print(f"{Colors.RED}Framework '{name}' not found{Colors.END}")
            return

        print_header(f"{fw['name']} - Detailed View")

        # Basic Info
        print_section("Basic Information")
        print(f"Type: {fw.get('type')}")
        print(f"Developer: {fw.get('developer')}")
        print(f"Language: {fw.get('language')}")
        print(f"First Release: {fw.get('first_release')}")
        print(f"License: {fw.get('license')}")
        print(f"GitHub Stars: ⭐ {fw.get('github_stars', 0):,}")

        # Use Cases
        print_section("Common Use Cases")
        for use_case in fw.get('use_cases', []):
            print(f"  • {use_case}")

        # Pros
        print_section(f"{Colors.GREEN}Advantages{Colors.END}")
        for pro in fw.get('pros', []):
            print(f"  ✓ {pro}")

        # Cons
        print_section(f"{Colors.YELLOW}Limitations{Colors.END}")
        for con in fw.get('cons', []):
            print(f"  ✗ {con}")

        # Deployment
        print_section("Deployment Tools")
        for tool in fw.get('deployment_tools', []):
            print(f"  • {tool}")

        # Market Info
        print_section("Market Information")
        print(f"Job Demand: {self._color_demand(fw.get('job_demand', 'Unknown'))}")
        print(f"Market Share: {fw.get('market_share', 'N/A')}")
        print(f"Learning Curve: {fw.get('learning_curve', 'N/A')}")

    def compare_frameworks(self, names: List[str]):
        """Compare multiple frameworks side-by-side."""
        frameworks = [self._find_framework(name) for name in names]
        frameworks = [fw for fw in frameworks if fw is not None]

        if len(frameworks) < 2:
            print(f"{Colors.RED}Need at least 2 valid frameworks to compare{Colors.END}")
            return

        print_header("Framework Comparison")

        # Basic comparison
        print_section("Basic Information")
        print(f"{'Attribute':<20} " + " | ".join(f"{fw['name']:<20}" for fw in frameworks))
        print("-" * (20 + 23 * len(frameworks)))

        attributes = [
            ('Type', 'type'),
            ('Developer', 'developer'),
            ('Language', 'language'),
            ('First Release', 'first_release'),
            ('License', 'license')
        ]

        for label, key in attributes:
            values = [fw.get(key, 'N/A')[:20] for fw in frameworks]
            print(f"{label:<20} " + " | ".join(f"{v:<20}" for v in values))

        # Market metrics
        print_section("Market Metrics")
        print(f"{'Metric':<20} " + " | ".join(f"{fw['name']:<20}" for fw in frameworks))
        print("-" * (20 + 23 * len(frameworks)))

        metrics = [
            ('GitHub Stars', 'github_stars'),
            ('Job Demand', 'job_demand'),
            ('Market Share', 'market_share'),
            ('Learning Curve', 'learning_curve')
        ]

        for label, key in metrics:
            values = []
            for fw in frameworks:
                val = fw.get(key, 'N/A')
                if isinstance(val, int):
                    val = f"{val:,}"
                values.append(str(val)[:20])
            print(f"{label:<20} " + " | ".join(f"{v:<20}" for v in values))

        # Recommendation
        print_section("Quick Recommendation")
        for fw in frameworks:
            demand = fw.get('job_demand', '')
            curve = fw.get('learning_curve', '')
            print(f"\n{Colors.BOLD}{fw['name']}{Colors.END}:")

            if 'Very High' in demand or 'High' in demand:
                print(f"  {Colors.GREEN}✓ Excellent job prospects{Colors.END}")
            if curve in ['Low', 'Very Low']:
                print(f"  {Colors.GREEN}✓ Beginner-friendly{Colors.END}")
            elif curve == 'High':
                print(f"  {Colors.YELLOW}⚠ Requires significant learning{Colors.END}")

            # Use case hint
            use_cases = fw.get('use_cases', [])[:2]
            if use_cases:
                print(f"  Best for: {', '.join(use_cases)}")

    def interactive_quiz(self):
        """Interactive quiz about ML frameworks."""
        print_header("ML Framework Knowledge Quiz")

        questions = [
            {
                "question": "Which framework is best known for research and has dynamic computation graphs?",
                "options": ["TensorFlow", "PyTorch", "scikit-learn", "XGBoost"],
                "answer": "PyTorch",
                "explanation": "PyTorch is favored in research due to its Pythonic API and dynamic computation graphs."
            },
            {
                "question": "Which is the go-to framework for traditional machine learning on tabular data?",
                "options": ["TensorFlow", "PyTorch", "scikit-learn", "JAX"],
                "answer": "scikit-learn",
                "explanation": "scikit-learn excels at traditional ML algorithms for classification, regression, and clustering."
            },
            {
                "question": "Which framework dominates the NLP/Transformers space?",
                "options": ["TensorFlow", "Hugging Face Transformers", "MXNet", "FastAI"],
                "answer": "Hugging Face Transformers",
                "explanation": "Hugging Face provides the largest library of pre-trained transformer models."
            },
            {
                "question": "For production deployment at Google, which framework would you likely use?",
                "options": ["PyTorch", "TensorFlow", "scikit-learn", "FastAI"],
                "answer": "TensorFlow",
                "explanation": "TensorFlow, developed by Google, has excellent production tooling like TF Serving."
            },
            {
                "question": "Which gradient boosting library is known for being the fastest on large datasets?",
                "options": ["XGBoost", "LightGBM", "CatBoost", "scikit-learn"],
                "answer": "LightGBM",
                "explanation": "LightGBM (by Microsoft) is optimized for speed and memory efficiency on large datasets."
            }
        ]

        score = 0
        for i, q in enumerate(questions, 1):
            print(f"\n{Colors.BOLD}Question {i}/{len(questions)}:{Colors.END}")
            print(f"{q['question']}\n")

            for idx, option in enumerate(q['options'], 1):
                print(f"  {idx}. {option}")

            while True:
                try:
                    answer = input(f"\n{Colors.CYAN}Your answer (1-4): {Colors.END}").strip()
                    choice_idx = int(answer) - 1
                    if 0 <= choice_idx < len(q['options']):
                        break
                    print(f"{Colors.RED}Please enter a number between 1 and 4{Colors.END}")
                except (ValueError, EOFError):
                    print(f"{Colors.RED}Invalid input{Colors.END}")
                    return

            chosen = q['options'][choice_idx]
            if chosen == q['answer']:
                print(f"{Colors.GREEN}✓ Correct!{Colors.END}")
                score += 1
            else:
                print(f"{Colors.RED}✗ Incorrect. The answer is: {q['answer']}{Colors.END}")

            print(f"{Colors.YELLOW}💡 {q['explanation']}{Colors.END}")

        # Final score
        percentage = (score / len(questions)) * 100
        print_header(f"Quiz Complete!")
        print(f"\nYour score: {Colors.BOLD}{score}/{len(questions)} ({percentage:.0f}%){Colors.END}\n")

        if percentage >= 80:
            print(f"{Colors.GREEN}Excellent! You have strong ML framework knowledge.{Colors.END}")
        elif percentage >= 60:
            print(f"{Colors.YELLOW}Good job! Keep learning about these frameworks.{Colors.END}")
        else:
            print(f"{Colors.RED}Review the ML frameworks and try again!{Colors.END}")

    def show_trends(self):
        """Show current industry trends."""
        print_header("ML Framework Trends (2024)")

        trends = self.frameworks_data.get('trends', {}).get('2024', {})

        print_section(f"{Colors.GREEN}Growing Frameworks{Colors.END}")
        for fw in trends.get('growing', []):
            print(f"  📈 {fw}")

        print_section(f"{Colors.BLUE}Stable Frameworks{Colors.END}")
        for fw in trends.get('stable', []):
            print(f"  ➡️  {fw}")

        print_section(f"{Colors.YELLOW}Declining Frameworks{Colors.END}")
        for fw in trends.get('declining', []):
            print(f"  📉 {fw}")

        # Focus areas
        print_section("Frameworks by Focus Area")
        focus = self.frameworks_data.get('trends', {}).get('focus_areas', {})
        for area, frameworks in focus.items():
            print(f"\n{Colors.BOLD}{area.upper()}:{Colors.END} {frameworks}")

    def _find_framework(self, name: str) -> Optional[Dict]:
        """Find framework by name (case-insensitive)."""
        name_lower = name.lower()
        for fw in self.frameworks:
            if fw['name'].lower() == name_lower:
                return fw
        return None

    def interactive_mode(self):
        """Run interactive exploration mode."""
        print_header("Welcome to ML Landscape Explorer!")
        print(f"{Colors.CYAN}Explore ML frameworks, compare options, and test your knowledge.{Colors.END}\n")

        while True:
            print(f"\n{Colors.BOLD}What would you like to do?{Colors.END}")
            print("1. List all frameworks")
            print("2. View framework details")
            print("3. Compare frameworks")
            print("4. Take a quiz")
            print("5. View industry trends")
            print("6. Exit")

            try:
                choice = input(f"\n{Colors.CYAN}Enter choice (1-6): {Colors.END}").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if choice == '1':
                self.list_frameworks()
            elif choice == '2':
                name = input(f"{Colors.CYAN}Enter framework name: {Colors.END}").strip()
                self.show_framework_details(name)
            elif choice == '3':
                names_str = input(f"{Colors.CYAN}Enter 2-3 framework names (comma-separated): {Colors.END}").strip()
                names = [n.strip() for n in names_str.split(',')]
                self.compare_frameworks(names)
            elif choice == '4':
                self.interactive_quiz()
            elif choice == '5':
                self.show_trends()
            elif choice == '6':
                print("\nGoodbye!")
                break
            else:
                print(f"{Colors.RED}Invalid choice{Colors.END}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Explore the AI/ML landscape interactively"
    )
    parser.add_argument(
        '--frameworks',
        action='store_true',
        help='List all frameworks'
    )
    parser.add_argument(
        '--details',
        type=str,
        metavar='NAME',
        help='Show details for specific framework'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        metavar='NAME',
        help='Compare frameworks (e.g., --compare PyTorch TensorFlow)'
    )
    parser.add_argument(
        '--quiz',
        action='store_true',
        help='Take a knowledge quiz'
    )
    parser.add_argument(
        '--trends',
        action='store_true',
        help='Show industry trends'
    )

    args = parser.parse_args()

    explorer = MLLandscapeExplorer()

    # Handle command-line arguments
    if args.frameworks:
        explorer.list_frameworks()
    elif args.details:
        explorer.show_framework_details(args.details)
    elif args.compare:
        explorer.compare_frameworks(args.compare)
    elif args.quiz:
        explorer.interactive_quiz()
    elif args.trends:
        explorer.show_trends()
    else:
        # Interactive mode
        explorer.interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
