#!/bin/bash
# Run a command on the measurement box with strict host-key checking (no bypass).
exec /usr/bin/ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -o ControlMaster=no -o ControlPath=none "${H:-bh-33}" "$@"
