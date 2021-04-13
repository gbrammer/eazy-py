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
	pip install . -r requirements.txt

    
Demonstration
~~~~~~~~~~~~~
See `HDFN-Example.ipynb <https://nbviewer.jupyter.org/github/gbrammer/eazy-py/blob/HEAD/docs/examples/HDFN-demo.ipynb>`__.