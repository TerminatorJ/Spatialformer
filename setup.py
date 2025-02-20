from setuptools import setup, find_packages


with open('./requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='spatialformer',  
    version='0.0.18',
    author='TerminatorJ',
    author_email='wangjun19950708@gmail.com',
    description='A single-cell foundation model focus on the spatial cell-cell colocalization',
    url='https://github.com/TerminatorJ/Spatialformer/', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
    ],
    python_requires='>=3.9', 
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'spatialformer.config': ['*.json'],  # Explicitly include JSON files
        'spatialformer.tokenizer': ['*.json'],
        'spatialformer.spatial_embeddings': ['*.pkl']
    },
    packages=find_packages(include=['spatialformer', 'spatialformer.*']),
   
)
