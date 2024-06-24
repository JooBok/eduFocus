FROM kubeflownotebookswg/jupyter-tensorflow-full

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    apt-utils \
  && usermod -aG sudo ${NB_USER} \
  && echo "${NB_USER}:${NB_USER}" | chpasswd \
  && echo "${NB_USER} ALL=(root) NOPASSWD:SETENV: /init" >> /etc/sudoers

# COPY requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt

USER jovyan
