from setuptools import setup, find_packages


with open('./requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='spatialformer',  
    version='0.0.12',
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
    packages=find_packages(include=['spatialformer', 'spatialformer.tools', 'spatialformer.model', 
                                    'spatialformer.model.submodules', 'spatialformer.config', 
                                    'spatialformer.tokenizer']),
   
)
