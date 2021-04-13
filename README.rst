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
See `HDFN-Example.ipynb <https://nbviewer.jupyter.org/github/gbrammer/eazy-py/blob/HEAD/docs/examples/HDFN-demo.ipynb>`__.