
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'MangetamainTESTWR'
copyright = '2025, William'
author = 'William'
release = '1.0'

# -- Path setup (permet d’importer le package src.*) -------------------------
import os
import sys
# docs/source/conf.py  →  ../../ = racine du repo, ../../src = code
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src'))

# -- Extensions ---------------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',          # <— nécessaire pour .. automodule::
    'sphinx.ext.napoleon',         # docstrings Google / NumPy
    'sphinx.ext.viewcode',         # liens vers le code
    'sphinx_autodoc_typehints',    # affiche/propulse les annotations de types
    'myst_parser',                 # support Markdown (README.md, etc.)
]

# Options utiles pour autodoc/napoleon
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST (Markdown)
myst_enable_extensions = ["linkify"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
templates_path = ['_templates']
exclude_patterns = []
language = 'fr'

# -- HTML --------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_rtd_theme'    # plus lisible que alabaster pour l’API
html_static_path = ['_static']

