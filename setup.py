from setuptools import setup

setup(
    name='SUMO-QL',
    version='1.1',
    packages=['sumo_ql', ],
    install_requires=[
        'numpy',
        'pandas',
        'gym',
        'ray[rllib]',
        'traci',
        'sumolib',
        'libsumo',
        'sklearn'
    ],
    author='guidytz',
    author_email='guidytz@gmail.com',
    url='https://github.com/guidytz/SUMO-QL',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description='A python code to handle Multi-agent Reinforcement Learning using SUMO for microscopic traffic routing.'
)
