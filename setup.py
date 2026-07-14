from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    install_requires = [line for line in f]

packages = [a for a in find_namespace_packages(where='.') if a[:7]=='mss_mpc']

setup(name = 'online_sysid',
      version = '0.0.2',
      description = 'Online learning of neural state-space models.',
      author = 'Bendeguz Gyorok',
      author_email = 'gyorokbende@sztaki.hu',
      python_requires = '>=3.10',
      packages = packages,
      install_requires = install_requires,
      extras_require = {
          "examples": ["matplotlib==3.10.7"],
          },
    )