.. image:: https://github.com/gbrammer/eazy-py/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/gbrammer/eazy-py/actions
    
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5012705.svg
   :target: https://doi.org/10.5281/zenodo.5012705
   

eazy-py: Pythonic photometric redshift tools based on EAZY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
Under heavy construction....

Documentation will be here: https://eazy-py.readthedocs.io/, though it's essentially just the module API for now.

Templates and filter files still here: https://github.com/gbrammer/eazy-photoz/.

.. note::
    Please submit any questions/comments/problems you have through the `Issues <https://github.com/gbrammer/eazy-py/issues>`_ interface.

Installation instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    cd /usr/local/share/python # or some other location
    
    ## # Fetch the old C-code repo to get the templates, priors, etc.
    ## git clone https://github.com/gbrammer/eazy-photoz.git

    # Fetch the eazy-py repo, the --recursive option fetches the photoz repo
    git clone --recurse-submodules https://github.com/gbrammer/eazy-py.git
    
    # use git clone --recurse for git<=2.12
    
    # Environment variable to point to the repo containing filter/template files
    echo "export EAZYCODE=${PWD}/eazy-py/eazy-photoz" >> ~/.bashrc
    
    # Build the python code and install it in your python env
    cd eazy-py
    pip install .  -r requirements.txt

Demo
~~~~

.. image:: https://colab.research.google.com/assets/colab-badge.svg
 :target: https://colab.research.google.com/github/gbrammer/eazy-py/blob/master/docs/examples/HDFN-demo.ipynb

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gbrammer/eazy-py/HEAD?filepath=docs%2Fexamples%2FHDFN-demo.ipynb
