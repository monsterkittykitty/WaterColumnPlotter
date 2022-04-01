# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# March 2022

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

VERSION = '0.0.1'
DESCRIPTION = 'Package for real-time water column data visualization.'

setup(
    name="WaterColumnPlotter",
    version=VERSION,
    author="Lynette Davis",
    author_email="ldavis@ccom.unh.edu",
    description=DESCRIPTION,
    url="https://github.com/monsterkittykitty/WaterColumnPlotter",
    keywords="hydrography ocean mapping sonar hydrographic survey water column",
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "numpy",
        "pandas",
        "pip",
        "PyQt5",
        "pyqtgraph",
        "python-dateutil",
        "pytz",
        "scipy",
        "setuptools",
        "six"
    ],
    entry_points={
        "gui_scripts": [
            "water_colum_plotter = WaterColumnPlotter.GUI.GUI_Main:main",
        ],
        "console_scripts": [
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)