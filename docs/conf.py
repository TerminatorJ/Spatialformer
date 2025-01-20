import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "SpatialFormer"
copyright = "2024, Ole Winther, Jun Wang"
author = "Jun Wang"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link",
    # "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# -- Options for nbshpinx ----------------------------------------------------
nbsphinx_execute = "never"