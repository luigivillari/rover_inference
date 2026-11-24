# Drive-JEPA inference service for the Jetson Orin Nano: ROS 2 + PyTorch WITH CUDA.
# Separate image and container from rover_collector (the collector image has no PyTorch).
#
# Jetson of the lab (2026-11-24): L4T R39.2.1 = JetPack 7.2, CUDA 13.2. From JetPack 7 the Jetson uses the
# standard ARM64 (SBSA) CUDA: PyTorch comes from the official pytorch.org wheels for CUDA 13.2 (cu132, CUDA
# libraries installed by pip); the old Jetson-only images (dustynv / l4t-pytorch, JetPack 6) do not work.
# The GPU driver comes from the host through the nvidia runtime (docker-compose: runtime nvidia).
#
# Two variants (build args, see docker-compose.yaml):
#   A (default) ros:humble-ros-base (Ubuntu 22.04, Python 3.10): same ROS as rover_collector. Needs a cu132
#               torch wheel for Python 3.10 (the newest torch releases start at 3.11: pip takes the newest
#               one that still has 3.10). If the build fails at the torch step -> variant B.
#   B           BASE_IMAGE=ros:jazzy-ros-base ROS=jazzy (Ubuntu 24.04, Python 3.12, the setup tested on the
#               NVIDIA forum). Jazzy and Humble talk over DDS for the standard messages used here
#               (CompressedImage, TransformStamped, Twist, String, Float64): check with `ros2 topic echo`.
# The build stops if the torch installed has no CUDA (a CPU torch would take seconds per prediction).
#
#   docker compose build                                                        # A
#   BASE_IMAGE=ros:jazzy-ros-base ROS=jazzy docker compose build                # B
ARG BASE_IMAGE=ros:humble-ros-base
FROM ${BASE_IMAGE}
ARG ROS=humble
ARG INSTALL_ROS=0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu132
ENV DEBIAN_FRONTEND=noninteractive ROS_DISTRO=${ROS} PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1

# only for a base image without ROS (e.g. a plain Ubuntu / L4T image)
RUN if [ "$INSTALL_ROS" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends curl gnupg lsb-release ca-certificates && \
      curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list && \
      apt-get update && apt-get install -y --no-install-recommends ros-${ROS}-ros-base && rm -rf /var/lib/apt/lists/*; \
    fi
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-numpy python3-yaml python3-opencv \
    && rm -rf /var/lib/apt/lists/*
# numpy of the system kept (python3-opencv is built against it): pip must not upgrade it
RUN python3 -c "import numpy; print(f'numpy=={numpy.__version__}')" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
# PyTorch WITH CUDA (large download: torch + nvidia-* CUDA libraries, several GB)
RUN pip3 install torch torchvision --index-url "$TORCH_INDEX" \
    && python3 -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda); assert torch.version.cuda, 'CPU-only torch: wrong index'"
# model code (rover_finetune / navsim_v1) needs timm + einops; --no-deps so pip never replaces the CUDA torch
RUN pip3 install --no-deps timm einops \
    && pip3 install huggingface_hub safetensors pytest mcap rosbags

WORKDIR /ws
COPY rover_inference /ws/rover_inference
COPY tools /ws/tools
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV PYTHONPATH=/ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "rover_inference.node", "--ros-args", "-p", "config:=/ws/config/inference.yaml"]
