#!/bin/bash

HOSTFILE="hostfile"
NODE_SCRIPT="/share/ifi/available-nodes.sh"

> "$HOSTFILE"

echo "Updating $HOSTFILE..."

"$NODE_SCRIPT" | while read -r node; do
    [ -z "$node" ] && continue

    if ssh -n -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$node" true 2>/dev/null; then
        echo "${node}" >> "$HOSTFILE"
        echo "Added ${node}"
    else
        echo "Warning: Could not reach $node" >&2
    fi
done

echo "Done! Updated $HOSTFILE successfully."