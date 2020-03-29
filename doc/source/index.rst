.. dbfp_anomaly documentation master file, created by
   sphinx-quickstart on Thu Sep  5 21:34:50 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to kuberspatiotemporal
==============================

A package for probabilistic, unparameterized
heterogeneous mixture models for risk analysis.
Probabilistic modeling has various advantages such as normalized confidence and no need for data normalization (just to name a few).
In this context,
unparameterized means that the normally required parameter :math:`k`, which indicates the number of components in the mixture,
is not necessary. The term infinite model is used synonymously. In [Kimura11]_ this type of model is also called *Dirichlet Process Mixture Model (DPM)*
due to the underlying probabilistic framework. Lastly, *heterogeneous* refers to mixed domains of the feature space, namely
allowing to learn simultaneously continuous and categorical features.

This work is primarily based on [Kimura11]_ (with the contributions of DeepMind's `Arnaud Doucet`_) and [Heinzl14]_ for
infinite (aka. unparameterized) modeling and incremental learning, and [Hunt94]_ for heterogeneous models.
Further details can be found in [Neal98]_ and [Krueger18]_ but they did not directly influence this work.

This package has been developed in 2020 by `Stefan Ulbrich <stefan.frank.ulbrich@gmail.com>`__ for Acceptto.

Contents
^^^^^^^^

.. toctree::
   :maxdepth: 2

   derivation
   discussion
   implementation
   todo
   appendix
   api/modules


Bibliography
^^^^^^^^^^^^


.. [Kimura11] : Kimura et al., *"Expectation-maximization algorithms for inference in Dirichlet processes mixture"*, 2011, `[PDF] <https://link.springer.com/article/10.1007/s10044-011-0256-4>`__

.. [Hunt94] : L. A. Hunt, *"Mixture Model Clustering of Data Set with Categorial and Continuous Variables"*, 1994, `[PDF] <http://www.thebookshelf.auckland.ac.nz/docs/NZOperationalResearch/conferenceproceedings/1994-proceedings/ORSNZ-proceedings-1994-49.pdf>`__

.. [Heinzl14] : F. Heinzl and G. Tutz, *"Additive mixed models with approximate Dirichlet process mixtures: the EM approach"*, `[Link] <https://link.springer.com/article/10.1007/s11222-014-9475-z>`__

.. [Krueger18] : Krueger et al., *"A Dirichlet Process Mixture Model of Discrete Choice"*, 2018, `[PDF] <https://arxiv.org/pdf/1801.06296.pdf>`__

.. [Neal98] : R. Neal and G. Hinton, *"A view of the EM algorithm that justifies incremental, sparse and other variants"*, 1998, `[Link] <https://www.cs.toronto.edu/~hinton/absps/emk.pdf>`__

.. _Arnaud Doucet: https://www.turing.ac.uk/people/researchers/arnaud-doucet

Indices and tables
^^^^^^^^^^^^^^^^^^


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
