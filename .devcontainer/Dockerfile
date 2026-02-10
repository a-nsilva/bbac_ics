# Imagem Base Oficial do ROS 2 Humble (Ubuntu 22.04)
FROM osrf/ros:humble-desktop-full

# Define variáveis de ambiente para evitar interatividade na instalação
ENV DEBIAN_FRONTEND=noninteractive

# 2. Instalar dependências básicas do sistema e ferramentas de build
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-rosdep \
    build-essential \
    cmake \
    git \
    # Limpeza para reduzir tamanho da imagem
    && rm -rf /var/lib/apt/lists/* 

# Inicializar rosdep
RUN rosdep init || true

# Cria a pasta do workspace (opcional)
WORKDIR /workspaces/bbac_ics

# Adiciona o 'source' ao .bashrc para não precisares digitar sempre
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /workspaces/bbac_ics/install/setup.bash" >> /root/.bashrc

# Comando padrão (mantém o container vivo)
CMD ["/bin/bash"]
