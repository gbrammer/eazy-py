Installation instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

Install with `pip`
==================
.. code:: bash

    pip install eazy

Install from the repository
===========================
.. code:: bash
    
    ### [OPTIONAL!] create a fresh conda environment
    conda create -n eazy39 python=3.9
    conda activate eazy39
    
    cd /usr/local/share/python # or some other location

    ### Fetch the eazy-py repo
    git clone https://github.com/gbrammer/eazy-py.git
    
    ### Build the python code
    cd eazy-py
    pip install . -r requirements.txt
    
    ### Install and run the test suite, which also downloads the templates and
    ### filters from the eazy-photoz repository if necessary
    pip install .[test] -r requirements.txt
    pytest
    
    
Binder Demo
~~~~~~~~~~~
.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gbrammer/eazy-py/HEAD?filepath=docs%2Fexamples%2FHDFN-demo.ipynb
