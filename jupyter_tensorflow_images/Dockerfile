FROM tensorflow/serving

ARG NB_USER=jovyan
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    apt-utils \
 && useradd -m ${NB_USER} \
 && echo "${NB_USER}:${NB_USER}" | chpasswd \
 && usermod -aG sudo ${NB_USER} \
 && echo "${NB_USER} ALL=(root) NOPASSWD:SETENV: /init" >> /etc/sudoers

# COPY requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt

USER ${NB_USER}