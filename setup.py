from setuptools import find_packages, setup

packages = find_packages()

setup(name="visual_behavior_ophys",
      version=0.1,
      description="visual behavior ophys analysis",
      author="marinag",
      author_email="marinag@alleninstitute.org",
      url="https://github.com/matchings/visual_behavior_ophys",
      packages=packages,
      requires=['pandas', 'numpy', 'scipy', 'seaborn'],
      include_package_data=True,
      package_data={
          "": ['*.png', '*.ico', '*.jpg', '*.jpeg'],
          },
      )
