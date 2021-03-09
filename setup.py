
from setuptools import setup

setup(
    name='hod_emu',
    version='0.1',
    description='HODEmu from Ragagnin et al. 2021' ,
    url='https://aragagnin.github.io',
    author='Antonio Ragagnin',
    author_email='antonio.ragagnin@inaf.it',
    license='GPLv3+',
#    packages=['.'],
    py_modules=["hod_emu","_hod_emu_sklearn_gpr_serialized.py"],
    zip_safe=False
)
