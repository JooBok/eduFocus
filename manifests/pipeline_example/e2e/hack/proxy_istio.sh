#!/usr/bin/env bash
set -euo pipefail

# stop all port-forward processes
trap ctrl_c INT

function ctrl_c() {
        echo "Stopping port-forward processes..."
        echo "Killing process $ISTIO_PID..."
        kill -KILL $ISTIO_PID
}

kubectl port-forward --address 0.0.0.0 -n istio-system svc/istio-ingressgateway 8080:443 &
ISTIO_PID=$!
echo "Started Istio port-forward, pid: $ISTIO_PID"
echo ISTIO_PID=$ISTIO_PID >> pids.env

sleep 1
