Installation instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    cd /usr/local/share/python # or some other location
    # Fetch the old C-code repo to get the templates, priors, etc.
    git clone https://github.com/gbrammer/eazy-photoz.git

    # Fetch the eazy-py repo
    git clone https://github.com/gbrammer/eazy-py.git
    
    # Build the python code
    cd eazy-py
    python setup.py install
    