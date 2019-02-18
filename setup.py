from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.md')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

version = {}
with open(os.path.join(_here, 'mdn', 'version.py')) as f:
    exec(f.read(), version)

setup(
    name='keras-mdn-layer',
    version=version['__version__'],
    description=('An MDN Layer for Keras using TensorFlow Probability.'),
    long_description=long_description,
    author='Charles Martin',
    author_email='charlepm@ifi.uio.no',
    url='https://github.com/cpmpercussion/keras-mdn-layer',
    license='MIT',
    packages=['mdn'],
    include_package_data=True,
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT Licese'],
    )
