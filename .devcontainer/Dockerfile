# Imagem Base Oficial do ROS 2 Humble (Ubuntu 22.04)
FROM osrf/ros:humble-desktop-full

# Define variáveis de ambiente para evitar interatividade na instalação
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

# Copia requirements.txt ANTES de instalar (melhor caching)
COPY requirements.lock /tmp/requirements.lock
#pip-compile requirements.txt --output-file=requirements.lock
RUN pip3 install --no-cache-dir -r /tmp/requirements.lock

# 2. Instalar dependências básicas do sistema e ferramentas de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    build-essential \
    cmake \
    python3-colcon-common-extensions \
    python3-rosdep \
    # Limpeza para reduzir tamanho da imagem
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install pip-tols \
    && pip3 install --no-cache-dir -r /tmp/requirements.txt

# Configurar o Ambiente ROS automaticamente

USER ros

# Adiciona o 'source' ao .bashrc para não precisares digitar sempre
RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc

# Cria a pasta do workspace (opcional)
WORKDIR /workspaces/bbac_ics

# Comando padrão (mantém o container vivo)
CMD ["/bin/bash"]
