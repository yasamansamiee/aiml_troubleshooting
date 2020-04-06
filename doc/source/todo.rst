TODOs and open issues
=====================

* Random reset is not yet extensively tested (preferable used for data changing over time)
* Improve incremental learning to better suite the situation at a client:
    * Rather than learning a sample at a time, allow for digesting "chunks" of data
    * These chunks should perform the more performant batch learning
* Optimization of the scaling factor (auto-tune)
* Improve documentation and warn about known sensitivity and possible instabilities of the algorithm, e. g.,
    * Numerical instabilities and when they can occur
    * Sensitivity to scaling factor
    * Treatment of degenerates
    * Pipeline setup needs to take care of matching number of components, dimensions
* Improve unit tests and examples (e.g., control test set and a chapter "how to use")
* Further experiments regarding scoring with integrals