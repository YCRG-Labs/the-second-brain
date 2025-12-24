#!/usr/bin/env python
"""Setup script for backward compatibility with older pip versions.

This file is provided for compatibility with older pip versions that don't
support pyproject.toml. For modern installations, pyproject.toml is used.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
