Ext-RESCAL
=================

Scalable Tensor Factorization
------------------------------

Ext-RESCAL is a memory efficient implementation of the [RESCAL algorithm](http://www.cip.ifi.lmu.de/~nickel/data/slides-icml2011.pdf). It is written in Python and relies on the SciPy Sparse module.

Current Version
------------
[0.1.3](https://github.com/nzhiltsov/Ext-RESCAL/archive/0.1.3.zip)

Features
------------

* 3-D sparse tensor factorization [1]
* Joint 3-D sparse tensor and 2-D sparse matrix factorization (extended version) [2]
* The implementation scales well to the domains with millions of nodes on the affordable hardware
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

1) Run the RESCAL algorithm to decompose a 3-D tensor with 2 latent components and zero regularization on the test data:

<pre>python rescal.py --latent 2 --lmbda 0 --input tiny-example --outputentities entity.embeddings.csv --log rescal.log</pre>

The test data set represents a tiny entity graph of 3 adjacency matrices (tensor slices) in the row-column representation. See the directory <i>tiny-example</i>.  Ext-RESCAL will output the latent factors for the entities into the file <i>entity.embeddings.csv</i>.

2) Run the extended version of RESCAL algorithm to decompose a 3-D tensor and 2-D matrix with 2 latent components and regularizer equal to 0.001 on the test data (entity graph and entity-term matrix):

<pre>python extrescal.py --latent 2 --lmbda 0.001 --input tiny-mixed-example --outputentities entity.embeddings.csv --outputterms term.embeddings.csv --log extrescal.log</pre>


Credit
----------------------

The original algorithms are an intellectual property of the authors of the cited papers.

Development and Contribution
----------------------

This is a fork of the original code base provided by [Maximilian Nickel](http://www.cip.ifi.lmu.de/~nickel/). Ext-RESCAL has been developed by [Nikita Zhiltsov](http://cll.niimm.ksu.ru/cms/lang/en_US/main/people/zhiltsov). Ext-RESCAL may contain some bugs, so, if you find any of them, feel free to contribute the patches via pull requests.


Disclaimer
---------------------
The author is not responsible for implications from the use of this software.

License
---------------------

Licensed under the GNU General Public License version 3 (GPLv3) ;
you may not use this work except in compliance with the License.
You may obtain a copy of the License in the LICENSE file, or at:

   http://www.gnu.org/licenses/gpl.html