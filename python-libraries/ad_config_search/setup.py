from setuptools import find_packages, setup

setup(
    # Metadata
    name='ad_config_search',
    version='0.0.1',
    url='https://github.com/EyalSel/AD-config-search',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],

    # Dependencies
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'plotly_express',
        'tqdm',
        'jsonlines',
        'pycocotools',
        'pathlib',
        'imageio',
        'jsonlines',
        'scikit-learn==0.23.2',  # just what's been tested
        'ray[tune]',
        'tune_sklearn',
        'seaborn',
        'fastparquet',
        'absl-py',
        'icecream',
        'more_itertools'
    ],

    # Contents
    packages=find_packages(exclude=["scripts*", "notebooks*"]),
)
