#!/bin/bash

# Gets the directory where this script exists
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pip install jupyterlab

## install nodejs, have to do it manually if sudo not available
# curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
# sudo apt-get install -y nodejs


conda install -c conda-forge nodejs -y

# install extensions
# https://github.com/aquirdTurtle/Collapsible_Headings
jupyter labextension install @aquirdturtle/collapsible_headings

# https://github.com/timkpaine/jupyterlab_templates
pip install jupyterlab_templates
jupyter labextension install jupyterlab_templates
jupyter serverextension enable --py jupyterlab_templates

# https://jupyterlab-code-formatter.readthedocs.io/en/latest/index.html
# jupyter labextension install @ryantam626/jupyterlab_code_formatter
# pip install jupyterlab_code_formatter
# jupyter serverextension enable --py jupyterlab_code_formatter

# https://github.com/jtpio/jupyterlab-theme-toggle
jupyter labextension install jupyterlab-theme-toggle
jupyter serverextension enable --py jupyterlab-theme-toggle

# https://github.com/deshaw/jupyterlab-execute-time
# jupyter labextension install jupyterlab-execute-time
# jupyter serverextension enable --py jupyterlab-execute-time

# https://github.com/youngthejames/jupyterlab_filetree
# jupyter labextension install jupyterlab_filetree
# jupyter serverextension enable --py jupyterlab_filetree

# more setup that's much easier to do in python
python "$DIR/install.py"
