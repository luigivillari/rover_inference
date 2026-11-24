# Drive-JEPA inference service for the Jetson Orin Nano: ROS 2 Humble + PyTorch WITH CUDA.
# Separate image and container from rover_collector (the collector image has no PyTorch).
#
# TO CHECK ON THE JETSON BEFORE THE FIRST BUILD:
#   cat /etc/nv_tegra_release ; dpkg -l | grep nvidia-jetpack     -> L4T / JetPack version
#   PyTorch must be the build WITH CUDA for that L4T: a CPU-only torch would take ~1-2 s per prediction
#   (the node refuses to start with device cuda if CUDA is missing).
# Two ways (build args, see docker-compose.yaml):
#   A. BASE_IMAGE = an L4T PyTorch image for this JetPack (NVIDIA NGC / Jetson AI Lab, Ubuntu 22.04)
#      + INSTALL_ROS=1 (ROS 2 Humble from the ROS apt repository)
#   B. BASE_IMAGE = ros:humble-ros-base + TORCH_INDEX = Jetson PyTorch wheel index for this JetPack
ARG BASE_IMAGE=ros:humble-ros-base
FROM ${BASE_IMAGE}
ARG INSTALL_ROS=0
ARG TORCH_INDEX=""
ENV DEBIAN_FRONTEND=noninteractive ROS_DISTRO=humble

RUN if [ "$INSTALL_ROS" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends curl gnupg lsb-release ca-certificates && \
      curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list && \
      apt-get update && apt-get install -y --no-install-recommends ros-humble-ros-base && rm -rf /var/lib/apt/lists/*; \
    fi
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-numpy python3-yaml python3-opencv \
    && rm -rf /var/lib/apt/lists/*
RUN if [ -n "$TORCH_INDEX" ]; then pip3 install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"; fi
# model code (rover_finetune / navsim_v1) needs timm + einops; --no-deps so pip never replaces the CUDA torch
RUN pip3 install --no-cache-dir --no-deps timm einops \
    && pip3 install --no-cache-dir huggingface_hub safetensors pytest mcap rosbags

WORKDIR /ws
COPY rover_inference /ws/rover_inference
COPY tools /ws/tools
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV PYTHONPATH=/ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "rover_inference.node", "--ros-args", "-p", "config:=/ws/config/inference.yaml"]
