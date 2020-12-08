Installation instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    cd /usr/local/share/python # or some other location

    ### Fetch the eazy-py repo
    # --recurse-submodules pulls the eazy-code repository attached as a 
    # submodule. If you're running git<2.1.13, use --recursive.  
    git clone --recurse-submodules https://github.com/gbrammer/eazy-py.git
    
    ### Build the python code
    cd eazy-py
	pip install . 
    # or python setup.py install
    
Demonstration
~~~~~~~~~~~~~
See `HDFN-Example.ipynb <https://github.com/gbrammer/eazy-py/blob/master/docs/examples/HDFN-demo.ipynb>`__.