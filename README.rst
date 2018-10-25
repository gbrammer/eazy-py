eazy-py: Pythonic photometric redshift tools based on EAZY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under heavy construction....

Documentation will be here: https://http://eazy-py.readthedocs.io/

Templates and filter files still here: https://github.com/gbrammer/eazy-photoz/

Installation instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    cd /usr/local/share/python # or some other location
    
    ## # Fetch the old C-code repo to get the templates, priors, etc.
    ## git clone https://github.com/gbrammer/eazy-photoz.git

    # Fetch the eazy-py repo, the --recursive option fetches the photoz repo
    git clone --recursive https://github.com/gbrammer/eazy-py.git
    
    # Environment variable to point to the repo containing filter/template files
    echo "export EAZYCODE=${PWD}/eazy-py/eazy-photoz" >> ~/.bashrc
    
    # Build the python code
    cd eazy-py
    python setup.py install
