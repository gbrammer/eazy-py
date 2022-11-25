Installation instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

Install with `pip`
==================
.. code:: bash

    pip install eazy
    
    # Forked dependencies that don't come with pip
    pip install git+https://github.com/gbrammer/dust_attenuation.git
    pip install git+https://github.com/gbrammer/dust_extinction.git

Install from the repository
===========================
.. code:: bash

    cd /usr/local/share/python # or some other location

    ### Fetch the eazy-py repo
    git clone https://github.com/gbrammer/eazy-py.git
    
    ### Build the python code
    cd eazy-py
    pip install . -r requirements.txt

    
Binder Demo
~~~~~~~~~~~
.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gbrammer/eazy-py/HEAD?filepath=docs%2Fexamples%2FHDFN-demo.ipynb