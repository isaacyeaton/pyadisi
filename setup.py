try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

DESCRIPTION = "A Python package fro animal locomotion studies"
LONG_DESCRIPTION = DESCRIPTION
NAME = "pyadisi"
AUTHOR = "Isaac J. Yeaton"
AUTHOR_EMAIL = "isaac.yeaton@vt.edu"
MAINTAINER = "Isaac J. Yeaton"
MAINTAINER_EMAIL = "isaac.yeaton@vt.edu"
DOWNLOAD_URL = 'http://github.com/isaacyeaton/pyadisi'
LICENSE = 'BSD 3-clause'
VERSION = '0.0.1'

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=DOWNLOAD_URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['pyadisi'],
     )