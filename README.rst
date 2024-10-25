.. image:: https://github.com/gbrammer/eazy-py/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/gbrammer/eazy-py/actions

.. image:: https://badge.fury.io/py/eazy.svg
    :target: https://badge.fury.io/py/eazy

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

    $ pip install eazy
    $ pip install eazy[vistool] # for the dash visualiation tool

    # Install extra dependency
    $ pip install git+https://github.com/karllark/dust_attenuation.git

    # Get templates and filters from https://github.com/gbrammer/eazy-photoz
    $ python -c "import eazy; eazy.fetch_eazy_photoz()"

Demo
~~~~

.. image:: https://colab.research.google.com/assets/colab-badge.svg
 :target: https://colab.research.google.com/github/gbrammer/eazy-py/blob/master/docs/examples/HDFN-demo.ipynb

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gbrammer/eazy-py/HEAD?filepath=docs%2Fexamples%2FHDFN-demo.ipynb


Citation
~~~~~~~~
Please cite both this repository and `Brammer et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJ...686.1503B/abstract>`_. A BiBTeX for this repository can be generated via the *Cite this repository* link in the upper left corner of the `GitHub page <https://github.com/gbrammer/eazy-py>`_.
