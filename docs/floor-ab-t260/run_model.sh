#!/bin/bash
cd /Users/smarton/dev/gstt2-wt/t260
for k in dev cpu_devset cpu; do
  T251_VARIANTS=dev,dev_pre156 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 docs/floor-ab-t260/t251/model.py docs/floor-ab-t260/out/model_$k docs/floor-ab-t260/out/model_in_$k.npz || exit 1
done
