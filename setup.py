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
    zip_safe=False,
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)