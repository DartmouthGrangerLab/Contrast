Contrast 
========


Contrast is an implementation of the contrast function described in:

.. topic:: Reference
	   
	   Visual crowding is predominantly visual contrast
	   (in submission)
	   A.M.Rodriguez, R.Granger
	   Dartmouth College

.. topic:: Abstract

	   Visual clutter affects our ability to see: objects that would be
	   identifiable on their own, may become unrecognizable when presented
	   close together (“crowding”) – but the psychophysical characteristics
	   of crowding have resisted simplification. Image properties initially
	   thought to produce crowding have paradoxically yielded unexpected
	   results, e.g., adding flanking objects can ameliorate crowding. The
	   resulting theory revisions have been sufficiently complex and
	   specialized as to make it difficult to discern what principles may
	   underlie the observed phenomena. A generalized formulation of simple
	   visual contrast is presented, arising from straightforward analyses of
	   center and surround neurons in the early visual stream. Extant
	   contrast measures, such as RMS contrast, are easily shown to fall out
	   as reduced special cases.  The new contrast metric surprisingly
	   predicts the principal findings of a broad range of crowding studies.
	   These crowding phenomena may thus be said to arise predominantly from
	   contrast, or are, at least, severely confounded by contrast
	   effects. (These findings may be distinct from accounts of other,
	   likely downstream, “cognitive” or “semantic” instances of crowding,
	   suggesting that at least two separate forms of crowding may resist
	   unification.) The new fundamental contrast formulation provides a
	   candidate explanatory framework that addresses multiple psychophysical
	   phenomena beyond crowding.

Install
-------

This program is written in Python 2.7.14 which includes pip 9.0.1.  https://www.python.org/

Ensure that the following Python packages are installed:
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

* numpy       numpy (1.14.0)
* scipy       scipy (1.0.0)
* Pillow      Pillow (4.3.0)  https://pillow.readthedocs.io/en/latest/installation.html
* matplotlib  matplotlib (2.1.1)
* pandas      pandas (0.22.0)

Install the latest version of these libraries::

  $ pip install numpy scipy Pillow matplotlib pandas

To run all the experiments from the paper::
  
  $ cd framework/experiments
  $ ./resetall
  $ ./runall
  $ ./reportall



