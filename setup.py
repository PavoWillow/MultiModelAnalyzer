from setuptools import setup, find_packages

setup(
    name='MultiModelAnalyzer',
    version='0.1.0',
    author='Patrick Loomis',
    author_email='pvwloomis@gmail.com',
    description='A multi-model analysis application using LaunchDarkly AI Configs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PavoWillow/MultiModelAnalyzer',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # List your dependencies here
        'pandas',
        'tiktoken',
        'openai',
        'anthropic',
        'google-generativeai',
        'nltk',
        'textstat',
        'ldclient',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)