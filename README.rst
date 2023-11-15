Accelerated large-scale image procesing in Python
-------------------------------------------------

A hands-on session presented by Joan Rue Queralt, with the collaboration of:

Matthieu Simeoni, Sepand Kashani, Thomas Debarre, Daniele Hamm and Salim Najib.

Content
-------
- `Presentation <https://github.com/joanrue/accel-large-image-proc-talk/blob/main/presentation.ipynb>`_: Slides.

- `Notebook 0 <https://github.com/joanrue/accel-large-image-proc-talk/blob/main/0-Data-preparation.ipynb>`_: Data preparation. Downloads Hubble space telescope data and chunks it into small images.
- `Notebook 1 <https://github.com/joanrue/accel-large-image-proc-talk/blob/main/1-Introduction-to-Numba.ipynb>`_: Introduction to Numba and JIT compilation.
- `Notebook 2 <https://github.com/joanrue/accel-large-image-proc-talk/blob/main/2-Introduction-to-Dask.ipynb>`_: Introduction to Dask and the dashboard.
- `Notebook 3 <https://github.com/joanrue/accel-large-image-proc-talk/blob/main/3-Introduction-to-Dask-Image.ipynb>`_: Introduciton to Dask-image and example of large-scale image processing.
- `Notebook 4 <https://github.com/joanrue/accel-large-image-proc-talk/blob/main/4-Application-Dask-Numba.ipynb>`_: Application of Dask + Numba: the structure tensor for feature extraction. 

Note
----

Before starting, please clone this repository and install depenencies as follows:

.. code-block:: bash

   $ git clone https://github.com/joanrue/accel-large-image-proc-talk
   $ cd accel-large-image-proc-talk/
   $ conda create -n accel_env python=3.11
   $ conda activate accel_env
   $ pip install jupyter
   $ conda install matplotlib scipy numba scikit-image dask distributed dask-image -c conda-forge
   $ jupyter-lab
