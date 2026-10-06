#!/bin/bash
# Use: ./filter_hosts.sh hostfile [number of cores] > <hostfile_cores>
HOSTFILE="$1"                                                     
CORES="$2" 
count=0                                                 
while read -r node _; do                                   
  [ -z "$node" ] && continue                                # skip if empty line
  info=$(ssh -n -o BatchMode=yes -o ConnectTimeout=3 "$node" "LC_ALL=C lscpu" 2>/dev/null) || continue  # english labels 
  tpc=$(echo "$info" | awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2); print $2}')   # Threads per core
  cps=$(echo "$info" | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); print $2}')   # cores per Socket
  soc=$(echo "$info" | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2); print $2}')            # Sockets
  if [ "$tpc" = "1" ] && [ $((cps * soc)) -eq "$CORES" ]; then   
    echo "$node slots=$CORES"                               # save nodes and no of slots
    count=$((count + 1))

  fi
done < "$HOSTFILE"                                               
echo "matched nodes: $count" >&2