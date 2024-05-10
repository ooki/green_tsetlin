# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
os.system("mkdir -p notebooks/ && cp -r ../examples/* notebooks/")
sys.path.insert(0, os.path.abspath('..'))


project = 'green_tsetlin'
copyright = "2024, Sondre 'Ooki' Glimsdal"
author = "Sondre 'Ooki' Glimsdal"
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'nbsphinx']

nbsphinx_execute = "never"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = ["ROOT"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme' 
html_theme = "renku"

html_logo = "_static/RTD_LOGO.png"
html_logo_width = 50  
html_logo_height = 50 

html_static_path = ['_static']
