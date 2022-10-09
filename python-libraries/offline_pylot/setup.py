from setuptools import find_packages, setup

setup(
    # Metadata
    name='offline_pylot',
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
        # 'pylot', the package pip requirements in pypl are broken
        'erdos',
        'tqdm',
        'jsonlines>=2.0.0',  # higher version needed for custom numpy encoding
        'pycocotools',
        'pathlib',
        'imageio',
        'jsonlines',
        'motmetrics'
    ],

    # Contents
    packages=find_packages(exclude=["scripts*", "notebooks*"]),
)
