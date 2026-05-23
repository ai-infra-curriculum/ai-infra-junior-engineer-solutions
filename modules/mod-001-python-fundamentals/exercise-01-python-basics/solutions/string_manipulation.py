#!/usr/bin/env python3
"""
String Manipulation and Text Processing

Comprehensive string operations, formatting, parsing, and text processing examples.
"""

import re
import string
from typing import List, Dict, Optional, Tuple
from collections import Counter


class StringOperations:
    """Basic string operations and transformations."""

    @staticmethod
    def basic_operations(text: str) -> Dict[str, any]:
        """
        Demonstrate basic string operations.

        Args:
            text: Input string

        Returns:
            Dictionary with various string operation results

        Example:
            >>> StringOperations.basic_operations("  Hello World  ")['stripped']
            'Hello World'
        """
        return {
            'length': len(text),
            'uppercase': text.upper(),
            'lowercase': text.lower(),
            'title_case': text.title(),
            'capitalize': text.capitalize(),
            'stripped': text.strip(),
            'left_strip': text.lstrip(),
            'right_strip': text.rstrip(),
            'is_alpha': text.isalpha(),
            'is_digit': text.isdigit(),
            'is_alphanumeric': text.isalnum(),
            'starts_with_hello': text.startswith('Hello'),
            'ends_with_world': text.endswith('world'),
        }

    @staticmethod
    def split_and_join(text: str, delimiter: str = ' ') -> Dict[str, any]:
        """
        Demonstrate splitting and joining operations.

        Args:
            text: Input string
            delimiter: Delimiter for splitting

        Returns:
            Dictionary with split/join results

        Example:
            >>> StringOperations.split_and_join("a,b,c", ",")['split']
            ['a', 'b', 'c']
        """
        words = text.split(delimiter)

        return {
            'split': words,
            'split_max_2': text.split(delimiter, maxsplit=2),
            'rsplit': text.rsplit(delimiter),
            'splitlines': text.splitlines(),
            'join_with_dash': '-'.join(words),
            'join_with_space': ' '.join(words),
        }

    @staticmethod
    def replace_operations(text: str) -> Dict[str, str]:
        """
        Demonstrate string replacement operations.

        Args:
            text: Input string

        Returns:
            Dictionary with replacement results

        Example:
            >>> StringOperations.replace_operations("hello world")['replace_hello']
            'hi world'
        """
        return {
            'replace_hello': text.replace('hello', 'hi'),
            'replace_all_spaces': text.replace(' ', '_'),
            'replace_first_only': text.replace('o', 'O', 1),
            'translate_digits': text.translate(str.maketrans('0123456789', 'XXXXXXXXXX')),
        }

    @staticmethod
    def search_operations(text: str, substring: str) -> Dict[str, any]:
        """
        Demonstrate string search operations.

        Args:
            text: Input string
            substring: String to search for

        Returns:
            Dictionary with search results

        Example:
            >>> StringOperations.search_operations("hello world", "world")['found']
            True
        """
        return {
            'found': substring in text,
            'count': text.count(substring),
            'find_index': text.find(substring),  # Returns -1 if not found
            'index': text.index(substring) if substring in text else None,  # Raises ValueError if not found
            'rfind': text.rfind(substring),  # Find from right
        }


class StringFormatting:
    """String formatting techniques."""

    @staticmethod
    def format_examples(name: str, age: int, balance: float) -> Dict[str, str]:
        """
        Demonstrate various string formatting techniques.

        Args:
            name: Person's name
            age: Person's age
            balance: Account balance

        Returns:
            Dictionary with formatted string examples

        Example:
            >>> result = StringFormatting.format_examples("Alice", 30, 1234.56)
            >>> "Alice" in result['f_string']
            True
        """
        return {
            'f_string': f"Name: {name}, Age: {age}, Balance: ${balance:.2f}",
            'format_method': "Name: {}, Age: {}, Balance: ${:.2f}".format(name, age, balance),
            'format_indexed': "Age: {1}, Name: {0}".format(name, age),
            'format_named': "Name: {n}, Age: {a}".format(n=name, a=age),
            'percent_style': "Name: %s, Age: %d, Balance: $%.2f" % (name, age, balance),
            'padding': f"{name:>20}",  # Right-aligned, width 20
            'zero_padding': f"{age:05d}",  # Zero-padded to 5 digits
        }

    @staticmethod
    def advanced_formatting(value: float) -> Dict[str, str]:
        """
        Demonstrate advanced number formatting.

        Args:
            value: Numeric value to format

        Returns:
            Dictionary with various number formats

        Example:
            >>> result = StringFormatting.advanced_formatting(1234567.89)
            >>> '1,234,567.89' in result['with_commas']
            True
        """
        return {
            'decimal_2': f"{value:.2f}",
            'decimal_4': f"{value:.4f}",
            'scientific': f"{value:e}",
            'percentage': f"{value:.2%}",
            'with_commas': f"{value:,.2f}",
            'with_sign': f"{value:+.2f}",
            'padded': f"{value:015.2f}",
        }


class TextProcessing:
    """Advanced text processing operations."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing extra whitespace and punctuation.

        Args:
            text: Input text

        Returns:
            Cleaned text

        Example:
            >>> TextProcessing.clean_text("  Hello,   world!  ")
            'Hello world'
        """
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()

    @staticmethod
    def word_count(text: str) -> int:
        """
        Count words in text.

        Args:
            text: Input text

        Returns:
            Number of words

        Example:
            >>> TextProcessing.word_count("Hello world, how are you?")
            5
        """
        return len(text.split())

    @staticmethod
    def word_frequency(text: str) -> Dict[str, int]:
        """
        Calculate word frequency in text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping words to their counts

        Example:
            >>> TextProcessing.word_frequency("hello world hello")
            {'hello': 2, 'world': 1}
        """
        # Clean and lowercase
        cleaned = TextProcessing.clean_text(text.lower())
        words = cleaned.split()
        return dict(Counter(words))

    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """
        Extract email addresses from text using regex.

        Args:
            text: Input text

        Returns:
            List of email addresses found

        Example:
            >>> TextProcessing.extract_emails("Contact: alice@example.com or bob@test.org")
            ['alice@example.com', 'bob@test.org']
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """
        Extract URLs from text using regex.

        Args:
            text: Input text

        Returns:
            List of URLs found

        Example:
            >>> TextProcessing.extract_urls("Visit https://example.com or http://test.org")
            ['https://example.com', 'http://test.org']
        """
        url_pattern = r'https?://[^\s]+'
        return re.findall(url_pattern, text)

    @staticmethod
    def extract_phone_numbers(text: str) -> List[str]:
        """
        Extract phone numbers from text.

        Args:
            text: Input text

        Returns:
            List of phone numbers found

        Example:
            >>> TextProcessing.extract_phone_numbers("Call 555-123-4567 or (555) 987-6543")
            ['555-123-4567', '(555) 987-6543']
        """
        # Matches formats: 555-123-4567, (555) 123-4567, etc.
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        return re.findall(phone_pattern, text)

    @staticmethod
    def remove_html_tags(html: str) -> str:
        """
        Remove HTML tags from text.

        Args:
            html: HTML string

        Returns:
            Text without HTML tags

        Example:
            >>> TextProcessing.remove_html_tags("<p>Hello <b>world</b></p>")
            'Hello world'
        """
        clean = re.sub(r'<[^>]+>', '', html)
        return ' '.join(clean.split())  # Clean up extra whitespace

    @staticmethod
    def is_palindrome(text: str) -> bool:
        """
        Check if text is a palindrome (ignoring spaces and case).

        Args:
            text: Input text

        Returns:
            True if palindrome, False otherwise

        Example:
            >>> TextProcessing.is_palindrome("A man a plan a canal Panama")
            True
        """
        # Remove non-alphanumeric and convert to lowercase
        cleaned = ''.join(c.lower() for c in text if c.isalnum())
        return cleaned == cleaned[::-1]

    @staticmethod
    def reverse_words(text: str) -> str:
        """
        Reverse the order of words in text.

        Args:
            text: Input text

        Returns:
            Text with reversed word order

        Example:
            >>> TextProcessing.reverse_words("Hello world from Python")
            'Python from world Hello'
        """
        return ' '.join(text.split()[::-1])

    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = '...') -> str:
        """
        Truncate text to maximum length.

        Args:
            text: Input text
            max_length: Maximum length
            suffix: Suffix to add when truncated

        Returns:
            Truncated text

        Example:
            >>> TextProcessing.truncate_text("This is a long sentence", 10)
            'This is...'
        """
        if len(text) <= max_length:
            return text

        truncated_length = max_length - len(suffix)
        return text[:truncated_length].rsplit(' ', 1)[0] + suffix


class StringValidation:
    """String validation functions."""

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address format.

        Args:
            email: Email address to validate

        Returns:
            True if valid format, False otherwise

        Example:
            >>> StringValidation.validate_email("test@example.com")
            True
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_password(password: str, min_length: int = 8) -> Tuple[bool, List[str]]:
        """
        Validate password strength.

        Args:
            password: Password to validate
            min_length: Minimum password length

        Returns:
            Tuple of (is_valid, list of failed requirements)

        Example:
            >>> valid, errors = StringValidation.validate_password("Weak")
            >>> valid
            False
        """
        errors = []

        if len(password) < min_length:
            errors.append(f"Password must be at least {min_length} characters")

        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")

        return len(errors) == 0, errors

    @staticmethod
    def is_valid_ip(ip: str) -> bool:
        """
        Validate IPv4 address format.

        Args:
            ip: IP address to validate

        Returns:
            True if valid IPv4, False otherwise

        Example:
            >>> StringValidation.is_valid_ip("192.168.1.1")
            True
        """
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if not re.match(pattern, ip):
            return False

        # Check each octet is 0-255
        octets = ip.split('.')
        return all(0 <= int(octet) <= 255 for octet in octets)


def parse_key_value_pairs(text: str, delimiter: str = '=', separator: str = ',') -> Dict[str, str]:
    """
    Parse key-value pairs from a string.

    Args:
        text: String containing key-value pairs
        delimiter: Delimiter between key and value
        separator: Separator between pairs

    Returns:
        Dictionary of parsed key-value pairs

    Example:
        >>> parse_key_value_pairs("name=Alice,age=30,city=NYC")
        {'name': 'Alice', 'age': '30', 'city': 'NYC'}
    """
    result = {}
    pairs = text.split(separator)

    for pair in pairs:
        if delimiter in pair:
            key, value = pair.split(delimiter, 1)
            result[key.strip()] = value.strip()

    return result


if __name__ == '__main__':
    print("=== String Manipulation Examples ===\n")

    # Basic operations
    text = "  Hello World  "
    ops = StringOperations()
    print("Basic Operations:")
    basic_result = ops.basic_operations(text)
    for key, value in basic_result.items():
        print(f"  {key}: {value}")

    # Formatting
    print("\nString Formatting:")
    formatter = StringFormatting()
    formats = formatter.format_examples("Alice", 30, 1234.56)
    for key, value in formats.items():
        print(f"  {key}: {value}")

    # Text processing
    print("\nText Processing:")
    processor = TextProcessing()

    sample_text = "Hello world! This is a test. Hello Python!"
    print(f"Word frequency: {processor.word_frequency(sample_text)}")
    print(f"Is palindrome ('A man a plan a canal Panama'): {processor.is_palindrome('A man a plan a canal Panama')}")

    email_text = "Contact us at support@example.com or sales@test.org"
    print(f"Extracted emails: {processor.extract_emails(email_text)}")

    # Validation
    print("\nValidation:")
    validator = StringValidation()
    print(f"Email valid ('test@example.com'): {validator.validate_email('test@example.com')}")
    print(f"IP valid ('192.168.1.1'): {validator.is_valid_ip('192.168.1.1')}")

    valid, errors = validator.validate_password("weak")
    print(f"Password valid: {valid}")
    if not valid:
        print(f"  Errors: {errors}")
