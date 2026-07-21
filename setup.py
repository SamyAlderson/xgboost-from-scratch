from setuptools import setup, find_packages

# Specify project metadata
NAME = 'xgboost-from-scratch'
VERSION = '1.0'

# Specify dependencies
DEPENDENCIES = [
    'numpy',
    'scipy',
    'pandas'
]

# Specify package structure
PACKAGES = find_packages('src')

# Specify package structure relative to this setup file
PACKAGE_DIR = {'': 'src'}

# Specify entry points
ENTRY_POINTS = {
    'console_scripts': [
        'xgboost = src.main:main',
    ],
}

setup(
    name=NAME,
    version=VERSION,
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    install_requires=DEPENDENCIES,
    entry_points=ENTRY_POINTS,
)