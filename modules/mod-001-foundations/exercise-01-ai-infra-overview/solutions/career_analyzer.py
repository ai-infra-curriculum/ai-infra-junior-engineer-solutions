#!/usr/bin/env python3
"""
Career Analyzer for AI Infrastructure Engineers

This tool helps you:
- Understand different AI infrastructure roles
- Assess your current skill level
- Identify skill gaps
- Create a personalized learning roadmap
- Explore career paths and salary ranges

Usage:
    python career_analyzer.py                    # Interactive mode
    python career_analyzer.py --roles            # List all roles
    python career_analyzer.py --assess           # Skill assessment
    python career_analyzer.py --roadmap          # Generate learning roadmap

Author: AI Infrastructure Curriculum Team
License: MIT
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


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


class CareerAnalyzer:
    """Main career analyzer class."""

    def __init__(self):
        """Initialize with career data."""
        self.roles_data = load_data('roles.json')
        if not self.roles_data:
            print(f"{Colors.RED}Failed to load roles data{Colors.END}")
            sys.exit(1)

        self.roles = self.roles_data.get('roles', [])
        self.skill_categories = self.roles_data.get('skill_categories', {})
        self.learning_paths = self.roles_data.get('learning_paths', {})

    def list_roles(self):
        """List all AI infrastructure roles."""
        print_header("AI Infrastructure Engineering Roles")

        for role in self.roles:
            salary = role.get('salary_range', {})
            sal_str = f"${salary.get('min', 0)//1000}k-${salary.get('max', 0)//1000}k"

            print(f"\n{Colors.BOLD}{Colors.GREEN}• {role['title']}{Colors.END}")
            print(f"  Level: {role.get('level')}")
            print(f"  Experience: {role.get('experience')}")
            print(f"  Salary Range: {sal_str} {salary.get('currency', 'USD')}")

            # Top skills
            prog_skills = role.get('required_skills', {}).get('programming', [])
            infra_skills = role.get('required_skills', {}).get('infrastructure', [])

            critical_skills = [
                s['skill'] for s in (prog_skills + infra_skills)
                if s.get('priority') == 'Critical'
            ][:5]

            if critical_skills:
                print(f"  Critical Skills: {', '.join(critical_skills)}")

            print(f"  Growth Path: {role.get('growth_path', 'N/A')}")

    def show_role_details(self, title: str):
        """Show detailed information about a specific role."""
        role = self._find_role(title)
        if not role:
            print(f"{Colors.RED}Role '{title}' not found{Colors.END}")
            print(f"\nAvailable roles:")
            for r in self.roles:
                print(f"  - {r['title']}")
            return

        print_header(f"{role['title']}")

        # Basic info
        print_section("Overview")
        print(f"Level: {role.get('level')}")
        print(f"Experience Required: {role.get('experience')}")

        salary = role.get('salary_range', {})
        print(f"Salary Range: ${salary.get('min', 0):,} - ${salary.get('max', 0):,} {salary.get('currency', 'USD')}")
        if salary.get('notes'):
            print(f"Note: {salary['notes']}")

        # Required skills
        print_section("Required Skills")
        skills = role.get('required_skills', {})

        for category in ['programming', 'infrastructure', 'ml_knowledge']:
            cat_skills = skills.get(category, [])
            if cat_skills:
                cat_name = category.replace('_', ' ').title()
                print(f"\n{Colors.BOLD}{cat_name}:{Colors.END}")
                for skill in cat_skills:
                    priority_color = {
                        'Critical': Colors.RED,
                        'Important': Colors.YELLOW,
                        'Helpful': Colors.GREEN
                    }.get(skill.get('priority', ''), '')

                    print(f"  • {skill['skill']} - {skill.get('proficiency', 'N/A')} "
                          f"({priority_color}{skill.get('priority', 'N/A')}{Colors.END})")

        # Soft skills
        soft_skills = skills.get('soft_skills', [])
        if soft_skills:
            print(f"\n{Colors.BOLD}Soft Skills:{Colors.END}")
            for skill in soft_skills:
                print(f"  • {skill}")

        # Typical tasks
        print_section("Typical Daily Tasks")
        for task in role.get('typical_tasks', []):
            print(f"  • {task}")

        # Day in the life
        day = role.get('day_in_life', {})
        if day:
            print_section("A Day in the Life")
            if day.get('morning'):
                print(f"\n{Colors.BOLD}Morning:{Colors.END}")
                for task in day['morning']:
                    print(f"  • {task}")
            if day.get('afternoon'):
                print(f"\n{Colors.BOLD}Afternoon:{Colors.END}")
                for task in day['afternoon']:
                    print(f"  • {task}")

        # Growth path
        print_section("Career Growth")
        print(f"Next Step: {role.get('growth_path', 'N/A')}")

    def skill_assessment(self):
        """Interactive skill assessment."""
        print_header("AI Infrastructure Skills Assessment")

        print(f"{Colors.CYAN}This assessment will help identify your skill gaps.{Colors.END}")
        print(f"{Colors.CYAN}Rate your proficiency: 1=Beginner, 2=Basic, 3=Intermediate, 4=Advanced, 5=Expert{Colors.END}\n")

        # Collect all unique skills from roles
        all_skills = {}
        for role in self.roles:
            req_skills = role.get('required_skills', {})
            for category in ['programming', 'infrastructure', 'ml_knowledge']:
                for skill in req_skills.get(category, []):
                    skill_name = skill['skill']
                    if skill_name not in all_skills:
                        all_skills[skill_name] = {
                            'category': category,
                            'importance': skill.get('priority', 'Helpful')
                        }

        # Assessment
        user_skills = {}
        print(f"{Colors.BOLD}Rate your proficiency (or press Enter to skip):{Colors.END}\n")

        for skill_name, info in sorted(all_skills.items()):
            try:
                rating = input(f"{skill_name} [{info['importance']}]: ").strip()
                if rating:
                    try:
                        rating_num = int(rating)
                        if 1 <= rating_num <= 5:
                            proficiency = ['Beginner', 'Basic', 'Intermediate', 'Advanced', 'Expert'][rating_num - 1]
                            user_skills[skill_name] = {
                                'rating': rating_num,
                                'proficiency': proficiency,
                                'category': info['category']
                            }
                    except ValueError:
                        pass
            except (EOFError, KeyboardInterrupt):
                print("\n\nAssessment interrupted.")
                return

        if not user_skills:
            print(f"\n{Colors.YELLOW}No skills rated. Assessment cancelled.{Colors.END}")
            return

        # Analyze results
        self._analyze_assessment(user_skills, all_skills)

    def _analyze_assessment(self, user_skills: Dict, all_skills: Dict):
        """Analyze skill assessment results."""
        print_header("Assessment Results")

        # Calculate readiness for each role
        print_section("Role Readiness")

        for role in self.roles:
            role_name = role['title']
            req_skills = role.get('required_skills', {})

            # Collect required skills
            critical_skills = []
            important_skills = []

            for category in ['programming', 'infrastructure', 'ml_knowledge']:
                for skill in req_skills.get(category, []):
                    skill_name = skill['skill']
                    required_prof = skill.get('proficiency', 'Basic')
                    priority = skill.get('priority', 'Helpful')

                    if priority == 'Critical':
                        critical_skills.append((skill_name, required_prof))
                    elif priority == 'Important':
                        important_skills.append((skill_name, required_prof))

            # Check how many critical skills are met
            critical_met = 0
            for skill_name, req_prof in critical_skills:
                if skill_name in user_skills:
                    # Simple proficiency check
                    user_prof = user_skills[skill_name]['proficiency']
                    if self._proficiency_level(user_prof) >= self._proficiency_level(req_prof):
                        critical_met += 1

            readiness = (critical_met / len(critical_skills) * 100) if critical_skills else 0

            # Display readiness
            color = Colors.GREEN if readiness >= 70 else Colors.YELLOW if readiness >= 40 else Colors.RED
            print(f"\n{Colors.BOLD}{role_name}{Colors.END}: {color}{readiness:.0f}% ready{Colors.END}")

            if readiness < 100:
                missing = [s for s, _ in critical_skills if s not in user_skills or
                          self._proficiency_level(user_skills[s]['proficiency']) <
                          self._proficiency_level(_)]
                if missing:
                    print(f"  Missing critical skills: {', '.join(missing[:3])}")

        # Skill gaps
        print_section("Recommended Focus Areas")

        # Categorize gaps
        gaps_by_category = {}
        for skill_name, info in all_skills.items():
            if skill_name not in user_skills or user_skills[skill_name]['rating'] < 3:
                category = info['category']
                if category not in gaps_by_category:
                    gaps_by_category[category] = []
                gaps_by_category[category].append(skill_name)

        for category, skills in gaps_by_category.items():
            cat_name = category.replace('_', ' ').title()
            print(f"\n{Colors.BOLD}{cat_name}:{Colors.END}")
            for skill in skills[:5]:  # Top 5
                print(f"  • {skill}")

    def _proficiency_level(self, proficiency: str) -> int:
        """Convert proficiency to numeric level."""
        levels = {
            'Fundamental': 1,
            'Beginner': 1,
            'Basic': 2,
            'Intermediate': 3,
            'Advanced': 4,
            'Expert': 5
        }
        return levels.get(proficiency, 0)

    def generate_roadmap(self, target_role: str = "Junior AI Infrastructure Engineer"):
        """Generate personalized learning roadmap."""
        print_header(f"Learning Roadmap: {target_role}")

        # Find the learning path
        path_key = 'beginner_to_junior' if 'Junior' in target_role else 'junior_to_mid'
        learning_path = self.learning_paths.get(path_key)

        if not learning_path:
            print(f"{Colors.RED}Learning path not found{Colors.END}")
            return

        print(f"{Colors.BOLD}Goal:{Colors.END} {learning_path['name']}")
        print(f"{Colors.BOLD}Duration:{Colors.END} {learning_path['duration']}")

        # Show phases
        print_section("Learning Phases")

        start_date = datetime.now()
        current_date = start_date

        for phase_info in learning_path.get('phases', []):
            phase_num = phase_info['phase']
            phase_name = phase_info['name']
            duration = phase_info['duration']

            print(f"\n{Colors.BOLD}{Colors.CYAN}Phase {phase_num}: {phase_name}{Colors.END}")
            print(f"Duration: {duration}")
            print(f"Start: {current_date.strftime('%B %Y')}")

            # Calculate end date (approximate)
            months = int(duration.split('-')[1].split()[0])
            end_date = current_date + timedelta(days=months * 30)
            print(f"Target End: {end_date.strftime('%B %Y')}")

            print(f"\n{Colors.BOLD}Focus Areas:{Colors.END}")
            for focus in phase_info['focus']:
                print(f"  • {focus}")

            current_date = end_date

        # Resources
        print_section("Recommended Resources")

        print(f"\n{Colors.BOLD}Programming:{Colors.END}")
        prog_skills = self.skill_categories.get('programming', {}).get('skills', [])
        for skill in prog_skills[:3]:
            print(f"\n{skill['name']} ({skill.get('learning_time', 'N/A')}):")
            for resource in skill.get('resources', [])[:2]:
                print(f"  • {resource}")

        print(f"\n{Colors.BOLD}Infrastructure:{Colors.END}")
        infra_skills = self.skill_categories.get('infrastructure', {}).get('skills', [])
        for skill in infra_skills[:3]:
            print(f"\n{skill['name']} ({skill.get('learning_time', 'N/A')}):")
            for resource in skill.get('resources', [])[:2]:
                print(f"  • {resource}")

        # Certifications
        print_section("Recommended Certifications")
        for cert in self.roles_data.get('certifications', [])[:4]:
            print(f"\n{Colors.BOLD}{cert['name']}{Colors.END}")
            print(f"  Provider: {cert['provider']}")
            print(f"  Relevance: {cert['relevance']}")
            print(f"  Cost: {cert['cost']}")
            print(f"  Prep Time: {cert.get('time_to_prepare', 'N/A')}")

    def show_market_insights(self):
        """Show job market insights."""
        print_header("AI Infrastructure Job Market Insights")

        market = self.roles_data.get('job_market', {})

        # Demand by location
        print_section("Demand by Location")
        for location, demand in market.get('demand_by_location', {}).items():
            color = Colors.GREEN if 'Very High' in demand else Colors.YELLOW if 'High' in demand else Colors.END
            print(f"  {location}: {color}{demand}{Colors.END}")

        # Company types
        print_section("By Company Type")
        for company_type, info in market.get('company_types', {}).items():
            type_name = company_type.replace('_', ' ').title()
            print(f"\n{Colors.BOLD}{type_name}{Colors.END}")
            print(f"  Examples: {', '.join(info.get('examples', [])[:3])}")
            print(f"  Salary: {info.get('salary_range', 'N/A')}")
            print(f"  Competition: {info.get('competition', 'N/A')}")
            print(f"  Interview: {info.get('interview_difficulty', 'N/A')}")

        # Trends
        print_section("2024 Trends")
        for trend in market.get('trends_2024', []):
            print(f"  📈 {trend}")

    def _find_role(self, title: str) -> Optional[Dict]:
        """Find role by title (case-insensitive partial match)."""
        title_lower = title.lower()
        for role in self.roles:
            if title_lower in role['title'].lower():
                return role
        return None

    def interactive_mode(self):
        """Run interactive career analysis."""
        print_header("Welcome to Career Analyzer!")
        print(f"{Colors.CYAN}Explore AI Infrastructure Engineering careers and plan your path.{Colors.END}\n")

        while True:
            print(f"\n{Colors.BOLD}What would you like to do?{Colors.END}")
            print("1. List all roles")
            print("2. View role details")
            print("3. Take skill assessment")
            print("4. Generate learning roadmap")
            print("5. View job market insights")
            print("6. Exit")

            try:
                choice = input(f"\n{Colors.CYAN}Enter choice (1-6): {Colors.END}").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if choice == '1':
                self.list_roles()
            elif choice == '2':
                title = input(f"{Colors.CYAN}Enter role title (e.g., 'Junior'): {Colors.END}").strip()
                self.show_role_details(title)
            elif choice == '3':
                self.skill_assessment()
            elif choice == '4':
                target = input(f"{Colors.CYAN}Target role [Junior AI Infrastructure Engineer]: {Colors.END}").strip()
                if not target:
                    target = "Junior AI Infrastructure Engineer"
                self.generate_roadmap(target)
            elif choice == '5':
                self.show_market_insights()
            elif choice == '6':
                print("\nGood luck on your career journey!")
                break
            else:
                print(f"{Colors.RED}Invalid choice{Colors.END}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and plan your AI infrastructure engineering career"
    )
    parser.add_argument(
        '--roles',
        action='store_true',
        help='List all roles'
    )
    parser.add_argument(
        '--details',
        type=str,
        metavar='ROLE',
        help='Show details for specific role'
    )
    parser.add_argument(
        '--assess',
        action='store_true',
        help='Take skill assessment'
    )
    parser.add_argument(
        '--roadmap',
        type=str,
        nargs='?',
        const='Junior AI Infrastructure Engineer',
        metavar='ROLE',
        help='Generate learning roadmap for role'
    )
    parser.add_argument(
        '--market',
        action='store_true',
        help='Show job market insights'
    )

    args = parser.parse_args()

    analyzer = CareerAnalyzer()

    # Handle command-line arguments
    if args.roles:
        analyzer.list_roles()
    elif args.details:
        analyzer.show_role_details(args.details)
    elif args.assess:
        analyzer.skill_assessment()
    elif args.roadmap:
        analyzer.generate_roadmap(args.roadmap)
    elif args.market:
        analyzer.show_market_insights()
    else:
        # Interactive mode
        analyzer.interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
