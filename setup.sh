#!/bin/bash

# Start Voilà on the assigned port, serving your notebook
voila hentsch_manifold_explorer.ipynb \
  --port=$PORT \
  --no-browser \
  --Voila.base_url="/" \
  --Voila.enable_nbextensions=True
