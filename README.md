Ext-RESCAL
=================

Scalable Tensor Factorization
------------------------------

Ext-RESCAL is a memory efficient implementation of [RESCAL](http://www.cip.ifi.lmu.de/~nickel/data/slides-icml2011.pdf), a state-of-the-art algorithm for DEDICOM-like tensor factorization. Ext-RESCAL is written in Python and relies on the SciPy Sparse module.

Current Version
------------
[0.5](https://github.com/nzhiltsov/Ext-RESCAL/archive/0.5.zip)

Features
------------

* 3-D sparse tensor factorization [1]
* Joint 3-D sparse tensor and 2-D sparse matrix factorization (extended version) [2]
* The implementation provably scales well to the domains with millions of nodes on the affordable hardware
* Handy input format

[1] M. Nickel, V. Tresp, H. Kriegel. A Three-way Model for Collective Learning on Multi-relational Data // Proceedings of the 28th International Conference on Machine Learning (ICML'2011). - 2011. 

[2] M. Nickel, V. Tresp, H. Kriegel. Factorizing YAGO: Scalable Machine Learning for Linked Data // Proceedings of the 21st international conference on World Wide Web (WWW'2012). - 2012.

Expected Applications
----------------------
* Link Prediction
* Collaborative Filtering
* Entity Search

Prerequisites
----------------------
* Python 2.7+
* Numpy 1.6+
* SciPy 0.12+

Usage Examples
----------------------

1) Let's imagine we have the following semantic graph:

![semantic-graph](tiny-mixed-example/semantic-graph.png)

Each tensor slice represents an adjacency matrix of the corresponding predicate (member-of, genre, cites). Run the RESCAL algorithm to decompose a 3-D tensor with 2 latent components and zero regularization on the test data:

<pre>python rescal.py --latent 2 --lmbda 0 --input tiny-example --outputentities entity.embeddings.csv --outputfactors latent.factors.csv --log rescal.log</pre>

The test data set represents a tiny entity graph of 3 adjacency matrices (tensor slices) in the row-column representation. See the directory <i>tiny-example</i>.  Ext-RESCAL will output the latent factors for the entities into the file <i>entity.embeddings.csv</i>.

2) Then, we assume that there is an entity-term matrix:

![entity-term-matrix](tiny-mixed-example/entity-term-matrix.png)

Run the extended version of RESCAL algorithm to decompose a 3-D tensor and 2-D matrix with 2 latent components and regularizer equal to 0.001 on the test data (entity graph and entity-term matrix):

<pre>python extrescal.py --latent 2 --lmbda 0.001 --input tiny-mixed-example --outputentities entity.embeddings.csv --outputterms term.embeddings.csv --outputfactors latent.factors.csv --log extrescal.log</pre>

If we plot the resulting embeddings, we would get the following picture, which reveals the similarity of entities and words in the latent space:

![latent-space-visualization](tiny-mixed-example/TinyMixedExample.png)

Development and Contribution
----------------------

Ext-RESCAL has been developed by [Nikita Zhiltsov](http://cll.niimm.ksu.ru/cms/lang/en_US/main/people/zhiltsov). This project is a fork of the original code base provided by [Maximilian Nickel](http://www.cip.ifi.lmu.de/~nickel/). Ext-RESCAL may contain some bugs, so, if you find any of them, feel free to contribute the patches via pull requests into the _develop_ branch.


Release Notes
------------

0.5 (March 29, 2013):

* Greatly improve the convergence speed via initialization of starting matrices with eigenvectors

0.4 (March 14, 2013):

* Add efficient computation of the exact objective value via trick with trace

0.3 (March 12, 2013):

* Fix random sampling for the basic task
* Add output of latent factors

0.2 (February 26, 2013): 

* Add an opportunity to approximate the objective function via random sampling
* Bug fixes
* Change the default settings

0.1 (January 31, 2013):

* The basic implementation of both the algorithms

Credit
----------------------

The original algorithms are an intellectual property of the authors of the cited papers.

Disclaimer
---------------------
The author is not responsible for implications from the use of this software.

License
---------------------

Licensed under the GNU General Public License version 3 (GPLv3) ;
you may not use this work except in compliance with the License.
You may obtain a copy of the License in the LICENSE file, or at:

   http://www.gnu.org/licenses/gpl.html