from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f]


DESC = """text to concepts similarity, based on the word-to-word similarity
esimtimates via the cosine-distance based on the word-embeddings.
"""

setup(
    name='t2csim',
    version='0.0.1',
    description=DESC,
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Text Processing :: Linguistic',
    ],
    url='https://github.com/eldrin/text-concept-similarity',
    author='Jaehun Kim',
    author_email='j.h.kim@tudelft.nl',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements(),
    test_suite='tests',
    entry_points={
        'console_scripts': ['t2csim=t2c.extract:main',
                            'gensim2hdf=t2c.word_embeddings:main'],
    },
    zip_safe=False
)
