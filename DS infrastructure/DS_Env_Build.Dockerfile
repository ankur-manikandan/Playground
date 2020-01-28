# We will use Ubuntu for our image
FROM ubuntu:16.04

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade && \
    apt-get install -y wget bzip2 git curl unzip vim htop

# Install Anaconda
RUN    wget https://repo.anaconda.com/archive/Anaconda2-5.3.0-Linux-x86_64.sh && \
    bash Anaconda2-5.3.0-Linux-x86_64.sh -b && \
    rm Anaconda2-5.3.0-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda2/bin:$PATH

# Updating Anaconda packages
#RUN conda update conda
#RUN conda update anaconda
#RUN conda update —all

# Upgrade pip
RUN pip install --upgrade pip \
    tensorflow \
    --upgrade tensorflow-probability

# Install Tensorflow
# RUN pip install tensorflow

# Install Tensorflow Probability
# RUN pip install --upgrade tensorflow-probability

# Install PyTorch
RUN conda install pytorch torchvision -c pytorch

# Install Pyro
RUN pip install pyro-ppl

# Configuring access to Jupyter
RUN mkdir /opt/notebooks
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

# Jupyter listens port: 8888
EXPOSE 8888

# Run Jupyter notebook on local machine
EXPOSE 8080

# Tensorboard
EXPOSE 6006

# Run Jupytewr notebook as Docker main process
#CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/opt/notebooks", "--ip='*'", #"--port=8888", "--no-browser"]

# Create a directory to mount the data_science volume to
RUN mkdir /ds

# Define working directory.
WORKDIR /ds

# Define default command.
CMD ["bash"]
