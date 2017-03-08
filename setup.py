import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="graph_tools",
    version="0.0.2",
    author="Alexander Belikov",
    author_email="abelikov@gmail.com",
    description="tools reducing bipartite graphs and applying ranking algorithms",
    license="BSD",
    keywords="eigenfactor",
    url="git@github.com:alexander-belikov/graph_tools.git",
    packages=['graph_tools'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 0 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=['numpy>=1.8.1',  'pandas>=0.17.0', 'networkx>=1.11']
)
