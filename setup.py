from setuptools import setup, find_packages


with open('./requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open("./README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='spatialformer',  
    version='0.1.8',
    author='TerminatorJ',
    author_email='wangjun19950708@gmail.com',
    description='A single-cell foundation model focus on the spatial cell-cell colocalization and subcellular mulecular co-occurrence',
    long_description=long_description,
    long_description_content_type="text/markdown",   # for Markdown
    url='https://github.com/TerminatorJ/Spatialformer/', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
    ],
    python_requires='>=3.10', 
    install_requires=requirements,
    extras_require={
        "simulation": ["sim-fish>=0.2.0", "umap==0.1.1"],
        "numba": ["torch_geometric>=2.5.3", "umap-learn>=0.5.4", "scanpy>=1.9.8"]
    },
    include_package_data=True,
    package_data={
        'spatialformer.config': ['*.json'],  # Explicitly include JSON files
        'spatialformer.tokenizer': ['*.json'],
        'spatialformer.spatial_embeddings': ['*.pkl'],
    },
    packages=find_packages(include=['spatialformer', 'spatialformer.*']),
   
)
