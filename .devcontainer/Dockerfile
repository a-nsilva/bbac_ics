# 1. Imagem Base Oficial do ROS 2 Humble (Ubuntu 22.04)
FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# 2. Instalar dependências básicas do sistema e ferramentas de build
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-setuptools \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/* 

RUN pip3 install --upgrade pip setuptools wheel

RUN rosdep init || true \
 && rosdep fix-permissions \
 && rosdep update

WORKDIR /workspaces/bbac_ics

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
# RUN echo "source /workspaces/bbac_ics/install/setup.bash" >> /root/.bashrc
RUN echo "if [ -f /workspaces/bbac_ics/install/setup.bash ]; then source /workspaces/bbac_ics/install/setup.bash; fi" >> /root/.bashrc

CMD ["/bin/bash"]
