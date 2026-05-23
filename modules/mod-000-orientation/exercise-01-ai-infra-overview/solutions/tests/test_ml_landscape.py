"""
Basic tests for ML Landscape Explorer and Career Analyzer

These tests verify that the core tools function correctly.
Uses only standard library to maintain no-external-dependencies design.
"""

import json
import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataFiles(unittest.TestCase):
    """Test that data files are valid and complete"""

    def setUp(self):
        """Set up test fixtures"""
        self.data_dir = Path(__file__).parent.parent / 'data'

    def test_frameworks_json_exists(self):
        """Test that frameworks.json exists and is valid"""
        frameworks_file = self.data_dir / 'frameworks.json'
        self.assertTrue(frameworks_file.exists(), "frameworks.json should exist")

        with open(frameworks_file, 'r') as f:
            data = json.load(f)

        self.assertIn('ml_frameworks', data)
        self.assertGreaterEqual(len(data['ml_frameworks']), 10,
                              "Should have at least 10 frameworks")

    def test_frameworks_structure(self):
        """Test that frameworks have required fields"""
        frameworks_file = self.data_dir / 'frameworks.json'

        with open(frameworks_file, 'r') as f:
            data = json.load(f)

        frameworks = data['ml_frameworks']
        required_fields = ['name', 'type', 'developer', 'language']

        for framework in frameworks:
            for field in required_fields:
                self.assertIn(field, framework,
                            f"{framework.get('name')} missing field: {field}")

    def test_frameworks_include_major_ones(self):
        """Test that major frameworks are included"""
        frameworks_file = self.data_dir / 'frameworks.json'

        with open(frameworks_file, 'r') as f:
            data = json.load(f)

        framework_names = [fw['name'] for fw in data['ml_frameworks']]

        # Check for major frameworks
        self.assertIn('PyTorch', framework_names)
        self.assertIn('TensorFlow', framework_names)
        self.assertIn('scikit-learn', framework_names)
        self.assertIn('XGBoost', framework_names)

    def test_roles_json_exists(self):
        """Test that roles.json exists and is valid"""
        roles_file = self.data_dir / 'roles.json'
        self.assertTrue(roles_file.exists(), "roles.json should exist")

        with open(roles_file, 'r') as f:
            data = json.load(f)

        self.assertIn('roles', data)
        self.assertEqual(len(data['roles']), 3,
                        "Should have exactly 3 role levels")

    def test_roles_structure(self):
        """Test that roles have required fields"""
        roles_file = self.data_dir / 'roles.json'

        with open(roles_file, 'r') as f:
            data = json.load(f)

        roles = data['roles']
        required_fields = ['title', 'level', 'experience', 'salary_range']

        for role in roles:
            for field in required_fields:
                self.assertIn(field, role,
                            f"{role.get('title')} missing field: {field}")

    def test_role_levels(self):
        """Test that all role levels are present"""
        roles_file = self.data_dir / 'roles.json'

        with open(roles_file, 'r') as f:
            data = json.load(f)

        role_titles = [role['title'] for role in data['roles']]

        self.assertIn('Junior AI Infrastructure Engineer', role_titles)
        self.assertIn('AI Infrastructure Engineer', role_titles)
        self.assertIn('Senior AI Infrastructure Engineer', role_titles)

    def test_salary_ranges_realistic(self):
        """Test that salary ranges are realistic"""
        roles_file = self.data_dir / 'roles.json'

        with open(roles_file, 'r') as f:
            data = json.load(f)

        for role in data['roles']:
            salary_range = role['salary_range']
            self.assertIn('min', salary_range)
            self.assertIn('max', salary_range)
            self.assertGreater(salary_range['max'], salary_range['min'],
                             f"{role['title']}: max should be greater than min")
            self.assertGreater(salary_range['min'], 50000,
                             "Min salary should be realistic")
            self.assertLess(salary_range['max'], 500000,
                          "Max salary should be realistic")

    def test_skill_categories_exist(self):
        """Test that skill categories are defined"""
        roles_file = self.data_dir / 'roles.json'

        with open(roles_file, 'r') as f:
            data = json.load(f)

        self.assertIn('skill_categories', data)

        categories = data['skill_categories']
        self.assertIn('programming', categories)
        self.assertIn('infrastructure', categories)
        self.assertIn('ml_knowledge', categories)

    def test_learning_paths_exist(self):
        """Test that learning paths are defined"""
        roles_file = self.data_dir / 'roles.json'

        with open(roles_file, 'r') as f:
            data = json.load(f)

        self.assertIn('learning_paths', data)

        paths = data['learning_paths']
        self.assertIn('beginner_to_junior', paths)

        beginner_path = paths['beginner_to_junior']
        self.assertIn('duration', beginner_path)
        self.assertIn('phases', beginner_path)
        self.assertGreater(len(beginner_path['phases']), 0,
                         "Learning path should have phases")

    def test_certifications_documented(self):
        """Test that certifications are documented"""
        roles_file = self.data_dir / 'roles.json'

        with open(roles_file, 'r') as f:
            data = json.load(f)

        self.assertIn('certifications', data)

        certs = data['certifications']
        self.assertGreater(len(certs), 0, "Should have some certifications")

        for cert in certs:
            self.assertIn('name', cert)
            self.assertIn('provider', cert)
            self.assertIn('cost', cert)

    def test_job_market_data_exists(self):
        """Test that job market data is present"""
        roles_file = self.data_dir / 'roles.json'

        with open(roles_file, 'r') as f:
            data = json.load(f)

        self.assertIn('job_market', data)

        job_market = data['job_market']
        self.assertIn('demand_by_location', job_market)
        self.assertIn('company_types', job_market)
        self.assertIn('trends_2024', job_market)


class TestMLLandscapeExplorer(unittest.TestCase):
    """Test the ML Landscape Explorer tool"""

    def setUp(self):
        """Set up test fixtures"""
        self.solutions_dir = Path(__file__).parent.parent
        self.explorer_script = self.solutions_dir / 'ml_landscape_explorer.py'

    def test_script_exists(self):
        """Test that ml_landscape_explorer.py exists"""
        self.assertTrue(self.explorer_script.exists())

    def test_script_is_executable(self):
        """Test that script can be imported"""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ml_landscape_explorer",
            self.explorer_script
        )
        module = importlib.util.module_from_spec(spec)

        # Should not raise an exception
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            # Script may call sys.exit(), which is ok
            pass


class TestCareerAnalyzer(unittest.TestCase):
    """Test the Career Analyzer tool"""

    def setUp(self):
        """Set up test fixtures"""
        self.solutions_dir = Path(__file__).parent.parent
        self.analyzer_script = self.solutions_dir / 'career_analyzer.py'

    def test_script_exists(self):
        """Test that career_analyzer.py exists"""
        self.assertTrue(self.analyzer_script.exists())

    def test_script_is_executable(self):
        """Test that script can be imported"""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "career_analyzer",
            self.analyzer_script
        )
        module = importlib.util.module_from_spec(spec)

        # Should not raise an exception
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            # Script may call sys.exit(), which is ok
            pass


class TestDocumentation(unittest.TestCase):
    """Test that documentation exists"""

    def setUp(self):
        """Set up test fixtures"""
        self.solutions_dir = Path(__file__).parent.parent

    def test_readme_exists(self):
        """Test that README.md exists"""
        readme = self.solutions_dir / 'README.md'
        self.assertTrue(readme.exists(), "README.md should exist")

    def test_readme_has_content(self):
        """Test that README has substantial content"""
        readme = self.solutions_dir / 'README.md'

        with open(readme, 'r') as f:
            content = f.read()

        # Should be substantial documentation
        self.assertGreater(len(content), 1000,
                         "README should have substantial content")

        # Should mention both tools
        self.assertIn('ML Landscape Explorer', content)
        self.assertIn('Career Analyzer', content)

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists"""
        requirements = self.solutions_dir / 'requirements.txt'
        self.assertTrue(requirements.exists(), "requirements.txt should exist")


if __name__ == '__main__':
    unittest.main()
