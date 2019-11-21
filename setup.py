from setuptools import setup, find_packages
import os

# Use Semantic Versioning, http://semver.org/
version_info = (0, 2, 0, '')
__version__ = '%d.%d.%d%s' % version_info


setup(name='mapanalysistools',
      version=__version__,
      description='Photostimulation Mapping Analysis Tools',
      url='https://',
      author='Paul B. Manis, Ph.D.',
      author_email='pmanis@med.unc.edu',
      license='MIT',
      packages=find_packages(include=['mapanalysistools*']),
      zip_safe=False,
      entry_points={
          'console_scripts': [
               'plotmaps=mapanalysistools.plot_maps:main',
          ]
      },
      classifiers = [
             "Programming Language :: Python :: 3.6+",
             "Development Status ::  Beta",
             "Environment :: Console",
             "Intended Audience :: Neuroscientists",
             "License :: MIT",
             "Operating System :: OS Independent",
             "Topic :: Scientific Software :: Tools :: Python Modules",
             "Topic :: Data Processing :: Neuroscience",
             ],
      )
      