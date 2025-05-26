#!/bin/bash

# Launch Voilà correctly for Render
voila hentsch_manifold_explorer.ipynb \
  --port="$PORT" \
  --no-browser \
  --Voila.base_url="/" \
  --Voila.ip=0.0.0.0
