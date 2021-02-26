from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='DL+DiReCT',
    version='0.9.1',
    description='DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SCAN-NRAD/DL-DiReCT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='brain morphometry, cortical thickness, MRI, neuroanatomy segmentation, deep learning',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.7',
    install_requires=['antspyx==0.2.4',
                      'more-itertools==8.0.0',
                      'nibabel==3.2.1',
                      'numpy==1.17.4',
                      'pandas==0.25.3',
                      'pyradiomics==3.0.1',
                      'scikit-learn==0.21.3',
                      'scikit-image==0.16.2',
                      'scipy==1.3.3',
                      'sortedcontainers==2.1.0',
                      'torch==1.3.1'],

    scripts=['dl+direct', 'direct'],
    project_urls={  # Optional
        #'Bug Reports': 'https://github.com/SCAN-NRAD/DL-DiReCT/issues',
        'Source': 'https://github.com/SCAN-NRAD/DL-DiReCT',
        'Publication': 'https://doi.org/10.1002/hbm.25159'
    },
)
