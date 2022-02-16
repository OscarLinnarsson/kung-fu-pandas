
from setuptools import setup

setup(
    name='kung_fu_pandas',
    version='0.0.1',
    description='A package providing abstractions of common pandas functions',
    
    url='https://github.com/OscarLinnarsson/kung-fu-pandas',
    author='Oscar Linnarsson',
    author_email='linnarsson.oscar@gmail.com',

    py_modules=['kung_fu_pandas'],
    install_requires=[
        'pandas',
        'numpy',
        'swifter',
    ],
)