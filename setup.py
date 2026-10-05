import re
from setuptools import setup

VERSIONFILE='juliet/_version.py'
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='juliet',
      version=verstr,
      description='juliet: a versatile modelling tool for transiting exoplanets, radial-velocity systems or both',
      url='http://github.com/nespinoza/juliet',
      author='Nestor Espinoza',
      author_email='nespinoza@stsci.edu',
      license='MIT',
      packages=['juliet', 'juliet.legacy'],
      install_requires=['jax>=0.4.31','jaxoplanet>=0.1.0','celerite2>=0.3.3','numpyro>=0.15','blackjax>=1.7','astropy','numpy','scipy','h5py'],
      python_requires='>=3.10',
      extras_require={
            'seaborn':['seaborn'],
            'matplotlib':['matplotlib'],
            'nautilus':['nautilus-sampler'],
            'legacy':['batman-package','radvel','george','celerite','dynesty','emcee','ultranest','zeus-mcmc','setuptools'],},
      entry_points={
            'console_scripts': [
                 'juliet=juliet.__main__:main'
            ]},
      zip_safe=False)
