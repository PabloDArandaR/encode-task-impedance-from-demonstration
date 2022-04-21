from setuptools import setup, find_packages

setup(name='pbdlib',
      version='0.1',
      description='Programming by Demonstration module for Python',
      url='',
      author='Emmanuel Pignat and Hakan Girgin',
      author_email='hakan.girgin@idiap.ch',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'matplotlib', 'sklearn', 'dtw', 'jupyter', 'termcolor'],
      zip_safe=False)
