from setuptools import setup, find_packages

setup(name="sentimentja",
      version="0.0.1f",
      description="Sentiment Analysis for Japanese",
      author="Shun Sugiyama",
      url="https://github.com/sugiyamath/sentiment_ja",
      packages=find_packages(),
      install_requires=[
          "keras==2.3.0",
          "tensorflow==2.3.1"
      ],
      package_data={
          'sentimentja':["*"]
      }
)
