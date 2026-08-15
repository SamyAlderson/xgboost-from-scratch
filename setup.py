from setuptools import setup, find_packages

NAME = 'xgboost-from-scratch'
VERSION = '1.0'

DEPENDENCIES = [
    'numpy',
    'scipy',
    'pandas'
]

PACKAGES = find_packages('src')

PACKAGE_DIR = {'': 'src'}

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
    include_package_data=True,
    zip_safe=False
)