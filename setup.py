# -*- encoding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlflow_sas",
    version="0.0.1",
    author="Aleksandra Gajowska",
    author_email="aleksandra.gajkowska@sas.com",
    description="MLFlow flavor for SAS Viya models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.sas.com/splakg/mlflow-sas-flavor",
    packages=setuptools.find_packages(include=['mlflow_sas', 'mlflow_sas.*']),
    entry_points={
        'console_scripts': [
            'model-upload=mmtools.command_line:model_upload',
            'get-model-uuid=mmtools.command_line:get_model_uuid',
        ],
    },
    install_requires=[
        'mlflow~=1.9', 'swat~=1.9'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False
)
