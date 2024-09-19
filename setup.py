# This file will be responsible of creating my machine learning application as a package and deploy them in PyPi

from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    ''''This function will read the requirements file and return a list of all the dependencies'''
    
    requirements = []
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements = [req.strip() for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
    
    name='mlproject',
    version='0.0.1',
    author='Mario Muñoz',
    author_email='mariomunozserrano@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt'),
    
    
)