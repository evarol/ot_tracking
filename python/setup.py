from setuptools import setup, find_packages

setup(
    name='otimage',
    version='0.0.1',
    description='OT-based signal extraction from calicum imaging data',
    long_description='(insert long description here)',
    author='Conor McGrory, Erdem Varol, Amin Nejatbaksh',
    author_email='cpmcgrory@gmail.com',
    url='https://github.com/evarol/ot_tracking/tree/master/python',
    license='',
    packages=find_packages(exclude=('tests', 'docs', 'notebooks', 'scripts'))
)
