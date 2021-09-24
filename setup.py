from setuptools import setup, find_packages

setup(name='conview',
      version='0.1',
      description='A specialized nifti viewer',
      author='Felix Knorr',
      author_email='knorr.felix@gmx.de',
      url='knorrfg.github.io',
      package_data={
        'conview.data': ['*']
      },
      packages=["conview", "conview.data"],
      install_requires=[
        "numpy", "nibabel", "pytest", 
          "nilearn", "click", "pyparadigm>=1.0.10",
          "matplotlib"
      ],
      entry_points={
          'console_scripts':[
              "conview = conview.conview:main"
              ]
          },
      license='MIT')
