# Discussion

## Data sources

The attributes that we are using are heterogeneous: For temporal and
spatiotemporal, i.e., location and time, these are continuous attributes (i.e.,
real numbers). They are not the only possible continuous attributes but
especially time is a crucial factor for it defines a behavior (as in sequence of
actions). For temporal and spatiotemporal behaviors, we use a stochastic mixture
model that deals well with this type of values.

### Categorical attributes

Categorical data is usually processed by anomaly/outlier detection
and classification algorithms but we use a heterogeneous model-driven approach.
Acceptto's AIML-backend features a newly introduced component of a REST-enabled API that
stores attributes/features of a fingerprint and computes basic feature
engineering / statistical analysis directly on the DB with SQL aggregations.
This technique is real-time capable thanks to sophisticated caching techniques.
The framework is not at all limited to fingerprinting attributes. On the
contrary, the code is dynamically generated and easily adjusted to specific use
cases including the ability to be configured to weight policies (it is not
limited to continuous values neither).

### Heterogeneous attributes
We use a custom
model (heterogenous, nonparametric probabilistic mixture model) to learn
behaviors from heterogeneous data (continuous and categorical). It divides the
data into a mixture of generative probability distributions. Historically, we
used this model to measure the performance of clustering/classification
algorithms and extended it by a proprietary learning method. The learning allows
for batch and incremental/online learning and does not depend on specialized
hardware acceleration and is faster compared to competing methods (especially
deep learning).


### Application

The real data has to fulfill the underlying hypothesis (inductive
bias) that valid users (and possibly threat actors too) can be grouped in way
supported by the probability distributions (have commonalities) and that
remaining threat actors are clear outliers or can be grouped themselves. If this
is not the case, alternative outlier detections methods are supposed as a
failback (at the expense of losing the beneficial properties of a probabilistic
approach and some of the supervised characteristics the method has)

Data set size: The minimal size of data to train a model is specific to the
client and determined by two factors:

1. is it for learning individual behavior or commonality analysis,
2. How much data is observed

Assuming the case of commonality analysis, a single day might give sufficient information to create
an initial model if there is a large user base. To be able to learn a behavior
that distinguishes between work days, a few instances of, for instance, Mondays
have to be observed naturally. The variance of the distributions in the mixture
is a good indicator of model quality.

Model adjustment: The frequency of computing the model is specific to the client
and determined by the same two factors:

1. is it for learning individual behavior or commonality analysis,
2. how much data is observed.

Hence, there is no general
answer as it strongly depends on the client’s ecosystem. For spatiotemporal
behavior modeling (mobile app) for instance, once a day turned out to be a good
value. Same holds for the amount of time data needs to be recorded until the
predictive quality is high. Idea: Note that the performance could be observed by
the MFA and Risk engine and the model can automatically be activated when it
performs well.

## From slides

* Data sets and sources

    * Data arrives from various sources and is heterogeneous
        * Continuous
        * Categorical
        * Each “event” yields a lot of information (“attributes”) from various sources (Event type, ThreatMatrix, Device fingerprints, timestamps, location, ..)


    * MFA results are unique
        * Interpreted as 3-state teaching signal (success, timeout, failure, nan)
        * Enables supervised learning
        * This data is incomplete (higher frequency early; occasional/random later)
* Attributes: Temporal and Spatiotemporal

    * Temporal and spatiotemporal
        * Continuous, cyclic domain
        * Ideal for describing behaviors
        * Pure temporal (1D): Learning single action
        * Spatiotemporal: Multidimensional, continuous domain time and location (continuous)
        * Additional discrete (categorical) features: day of week, surrounding WIFI

    * Individual vs. commonality
        * High-volumetric data required
        * Verified individual or Commonality analysis (grouping) Individual behavior
        * Assumption: Distribution over time forms clusters

* Attributes: Categorical
    * Multivariate Categorical attributes
        * Device fingerprinting: Language settings, OS type, etc.
        * Traditional preprocessing: One-hot encoding
        * Outlier/Anomaly detection / Classification


    * REST-enabled API / Microservice
        * For attribute storage, preprocessing and statistical analysis
        * Processing of categorical attributes (Not limited to fingerprint data)
        * Computations optimized for database efficiency (aggregations) and caching
        * Outlier detection & behavior learning implemented modularized (preprocessing & pipelines, for simple integration * of custom algorithms)
        * Connection to publish-subscribe service / Data Hub
        * Easy deployment and scalability

* Attributes: From spatiotemporal to heterogeneous data

    * Heterogeneous Data
        * Continuous / Cyclic / ordinal .. (e.g., locations, time, counts)
        * Pure categorical (e.g., most fingerprint attributes)
        * Incomplete data (i.e., MFA as an approximate teaching signal)


    * Underlying assumption / bias
        * Data is high in volume but sparse per user
        * Observed behaviors of many users contains communalities
        * Users (and attackers) can be grouped together (finite number of groups << data points)
        * Outliers (ungroupable observations) are considered attackers

* Heterogenous, nonparametric probabilistic mixture model

    * Models “acceptable” attribute values as probability distributions per group
    * Automatic deduction of groups, explainable and extractable representation
    * In general, unsupervised learning with supervised characteristics (teaching signal)
    * Probabilistic inference gives probabilistic answer (with confidence).
    * “Sharpness” of distributions is indicator for quality (manually, except for teaching signal)
    * Generative model: Verification of other algorithms
    * Customized/proprietary learning algorithm (fast, batch and incremental/online)