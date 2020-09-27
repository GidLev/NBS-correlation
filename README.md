# NBS-correlation
Python implementation of Network-based statistic for correlation testing

Network-Based Statistic procedure (NBS; Zalesky, Fornito, & Bullmore, 2010) is a common method for dealing with multiple comparisons in whole brain network analysis. Specifically, this method looks for subnetworks whose edges differ across experimental groups or are correlated with specific measured variables (e.g. weight loss in the current work).
Bctpy (https://pypi.org/project/bctpy/), the python brain connectivity toolbox (https://sites.google.com/site/bctnet/) excellent implementation, only cover the first case of two group connectivity comparison. The code here is modified for detecting set of edges correlated with an external variable. Another change from the original code is that only edges of the upper triangular are tested for efficiency, hence this function is only relevant for symmetric matrices. 

This is a modified version of the Bctpy (https://pypi.org/project/bctpy/) NBS implementation. 

Credit for implementing the vectorized version of the code to Gideon Rosenthal.

## Prerequities

* python > 2.7
* Bctpy (https://pypi.org/project/bctpy/)


## Usage

pvals, adj, null = nbs_bct_corr_z(subs_functional_conn_mats, thresh = 0.25, y_vec = weight_loss_vec, k=10000 ,verbose=False) 


## Refernce

* Levakov, G., Kaplan, A., Yaskolka M. A., Tsaban, G., Zelicha, H., Meiran, N., Shelef, I., Shai, I. & Avidan, G. (in press). Neural correlates of future weight loss reveal a possible role for brain-gastric interactions. NeuroImage.
