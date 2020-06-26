.. Contrast documentation master file, created by
   sphinx-quickstart on Wed Jan  2 15:26:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _contents:

Contrast documentation contents
===============================

.. contents::
   :depth: 2
  

.. toctree::
   :maxdepth: 2

	      
   README.rst
   overview
   reference/index


..
   Contrast Docs
   ===============

   Python 2.7.14  includes pip 9.0.1  https://www.python.org/

   Ensure that the following Python packages are installed:
   --------------------------------------------------------

   * numpy       numpy (1.14.0)
   * scipy       scipy (1.0.0)
   * Pillow      Pillow (4.3.0)  https://pillow.readthedocs.io/en/latest/installation.html
   * matplotlib  matplotlib (2.1.1)
   * pandas      pandas (0.22.0)::

       pip install numpy scipy Pillow matplotlib pandas

   To run::

     cd framwork/experiments
     ./resetall
     ./runall
     ./reportall


   .. _formatting-text:

   Formatting text
   ===============

   You use inline markup to make text *italics*, **bold**, or ``monotype``.


   .. _making-a-list:

   Making a list
   =============

   It is easy to make lists in rest

   Bullet points
   -------------

   This is a subsection making bullet points

   * point A

   * point B

   * point C


   Enumerated points
   ------------------

   This is a subsection making numbered points

   #. point A

   #. point B

   #. point C


   .. _making-a-table:

   Making a table
   ==============

   This shows you how to make a table -- if you only want to make a list see :ref:`making-a-list`.

   ==================   ============
   Name                 Age
   ==================   ============
   John D Hunter        40
   Cast of Thousands    41
   And Still More       42
   ==================   ============

   Here is something I want to talk about::

       def my_fn(foo, bar=True):
	   """A really useful function.

	   Returns None
	   """


   which is rendered as

   .. math::

      W^{3\beta}_{\delta_1 \rho_1 \sigma_2} \approx U^{3\beta}_{\delta_1 \rho_1}
   
