# Imagem Base Oficial do ROS 2 Humble (Ubuntu 22.04)
FROM osrf/ros:humble-desktop-full

# Define variáveis de ambiente para evitar interatividade na instalação
ENV DEBIAN_FRONTEND=noninteractive

# 2. Instalar dependências básicas do sistema e ferramentas de build
#RUN apt-get update && apt-get install -y --no-install-recommends \
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    build-essential \
    cmake \
    
    # Limpeza para reduzir tamanho da imagem
    && rm -rf /var/lib/apt/lists/* 

# Usuário vscode
#ARG USERNAME=vscode
#ARG USER_UID=1000
#ARG USER_GID=$USER_UID

#RUN groupadd --gid $USER_GID $USERNAME \
#    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#    && apt-get update \
#    && apt-get install -y sudo \
#    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#    && chmod 0440 /etc/sudoers.d/$USERNAME

# Cria a pasta do workspace (opcional)
WORKDIR /workspaces/bbac_ics

# Copia requirements.txt ANTES de instalar (melhor caching)
COPY requirements.txt /tmp/requirements.txt

#pip-compile requirements.txt --output-file=requirements.lock
RUN pip3 install -r /tmp/requirements.txt

# Adiciona o 'source' ao .bashrc para não precisares digitar sempre
RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc

# Comando padrão (mantém o container vivo)
CMD ["/bin/bash"]
