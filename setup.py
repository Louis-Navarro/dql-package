from setuptools import setup, find_packages

setup(
    name='dql-agent',
    version=1.0,
    description='A pre-made DQL Agent.',
    url='https://github.com/Louis-Navarro/dql-package',
    author='Louis Navarro',
    author_email='',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'tensorflow'],
    python_requires='>=3.*.*',
    classifiers=[
        'Programming Language :: Python :: 3'
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent'
    ],
)
