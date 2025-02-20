FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04


#Install Conda
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 

ENV PATH="/opt/conda/bin:${PATH}"




WORKDIR /home/
COPY spatialformer spatialformer
COPY setup.py .
COPY README.md .
COPY requirements.txt .

RUN pip install setuptools wheel
RUN python setup.py sdist bdist_wheel
RUN pip install twine
ARG PYPI_TOKEN
RUN twine upload dist/* -u __token__ -p $PYPI_TOKEN
RUN conda install python=3.9
RUN pip install SpatialFormer --extra-index-url https://download.pytorch.org/whl/cu121
RUN python -c "import spatialformer"