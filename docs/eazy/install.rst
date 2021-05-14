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

    
Binder Demo
~~~~~~~~~~~
.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gbrammer/eazy-py/HEAD?filepath=docs%2Fexamples%2FHDFN-demo.ipynb