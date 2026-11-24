#!/bin/bash
# Source ROS 2, then run the command (default: the inference node).
set -e
source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
exec "$@"
