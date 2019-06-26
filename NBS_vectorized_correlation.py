from __future__ import division
import numpy as np
from scipy import stats
from bct.algorithms import get_components

def nbs_bct_corr_z(corr_arr, thresh, y_vec, k=1000, extent=True, verbose=False):

    '''
    Performs the NBS for populations X and Y for a t-statistic threshold of
    alpha.

    Parameters
    ----------
    corr_arr : NxNxP np.ndarray
        matrix representing the correlation matrices population with P subjects. must be
        symmetric.

    y_vec : 1xP vector representing the behavioral/physiological values to correlate against

    thresh : float
        minimum Pearson's r-value used as threshold
    k : int
        number of permutations used to estimate the empirical null
        distribution, recommended - 10000
    verbose : bool
        print some extra information each iteration. defaults value = False

    Returns
    -------
    pval : Cx1 np.ndarray
        A vector of corrected p-values for each component of the networks
        identified. If at least one p-value is less than alpha, the omnibus
        null hypothesis can be rejected at alpha significance. The null
        hypothesis is that the value of the connectivity from each edge has
        equal mean across the two populations.
    adj : IxIxC np.ndarray
        an adjacency matrix identifying the edges comprising each component.
        edges are assigned indexed values.
    null : Kx1 np.ndarray
        A vector of K sampled from the null distribution of maximal component
        size.

    Notes
    -----
    ALGORITHM DESCRIPTION
    The NBS is a nonparametric statistical test used to isolate the
    components of an N x N undirected connectivity matrix that differ
    significantly between two distinct populations. Each element of the
    connectivity matrix stores a connectivity value and each member of
    the two populations possesses a distinct connectivity matrix. A
    component of a connectivity matrix is defined as a set of
    interconnected edges.

    The NBS is essentially a procedure to control the family-wise error
    rate, in the weak sense, when the null hypothesis is tested
    independently at each of the N(N-1)/2 edges comprising the undirected
    connectivity matrix. The NBS can provide greater statistical power
    than conventional procedures for controlling the family-wise error
    rate, such as the false discovery rate, if the set of edges at which
    the null hypothesis is rejected constitues a large component or
    components.

    The NBS comprises fours steps:
    1. Perform a Pearson r test at each edge indepedently to test the
       hypothesis that the value of connectivity between each edge and an
       external variable, corelates across all nodes.
    2. Threshold the Pearson r-statistic available at each edge to form a set of
       suprathreshold edges.
    3. Identify any components in the adjacency matrix defined by the set
       of suprathreshold edges. These are referred to as observed
       components. Compute the size of each observed component
       identified; that is, the number of edges it comprises.
    4. Repeat K times steps 1-3, each time randomly permuting the extarnal
       variable vector and storing the size of the largest component
       identified for each permutation. This yields an empirical estimate
       of the null distribution of maximal component size. A corrected
       p-value for each observed component is then calculated using this
       null distribution.

    [1] Zalesky A, Fornito A, Bullmore ET (2010) Network-based statistic:
        Identifying differences in brain networks. NeuroImage.
        10.1016/j.neuroimage.2010.06.041

     Adopted from the python implementation of the BCT - https://sites.google.com/site/bctnet/, https://pypi.org/project/bctpy/
     Credit for implementing the vectorized version of the code to Gideon Rosenthal

    '''

    def corr_with_vars(x, y):
        # check correlation X -> M (Sobel's test)
        r, _ = stats.pearsonr(x, y)
        z = 0.5 * np.log((1 + r)/(1 - r))
        return np.asscalar(z)

    ix, jx, nx = corr_arr.shape
    ny, = y_vec.shape

    if not ix == jx:
        raise ValueError('Matrices are not symmetrical')
    else:
        n = ix

    if nx != ny:
        raise ValueError('The [y_vec dimension must match the [corr_arr] third dimension')

    # only consider upper triangular edges
    ixes = np.where(np.triu(np.ones((n, n)), 1))

    # number of edges
    m = np.size(ixes, axis=1)

    # vectorize connectivity matrices for speed
    xmat = np.zeros((m, nx))

    for i in range(nx):
        xmat[:, i] = corr_arr[:, :, i][ixes].squeeze()
    del corr_arr

    # perform pearson corr test at each edge

    z_stat = np.apply_along_axis(corr_with_vars, 1, xmat, y_vec)
    print('z_stat: ', z_stat)

    # threshold
    ind_r, = np.where(z_stat > thresh)

    if len(ind_r) == 0:
        raise ValueError("Unsuitable threshold")

    # suprathreshold adjacency matrix
    adj = np.zeros((n, n))
    adjT = np.zeros((n, n))

    if extent:
        adj[(ixes[0][ind_r], ixes[1][ind_r])] = 1
        adj = adj + adj.T  # make symmetrical
    else:
        adj[(ixes[0][ind_r], ixes[1][ind_r])] = 1
        adj = adj + adj.T  # make symmetrical
        adjT[(ixes[0], ixes[1])] = z_stat
        adjT = adjT + adjT.T  # make symmetrical
        adjT[adjT <= thresh] = 0

    a, sz = get_components(adj)

    # convert size from nodes to number of edges
    # only consider components comprising more than one node (e.g. a/l 1 edge)
    ind_sz, = np.where(sz > 1)
    ind_sz += 1
    nr_components = np.size(ind_sz)
    sz_links = np.zeros((nr_components,))
    for i in range(nr_components):
        nodes, = np.where(ind_sz[i] == a)
        if extent:
            sz_links[i] = np.sum(adj[np.ix_(nodes, nodes)]) / 2
        else:
            sz_links[i] = np.sum(adjT[np.ix_(nodes, nodes)]) / 2

        adj[np.ix_(nodes, nodes)] *= (i + 2)

    # subtract 1 to delete any edges not comprising a component
    adj[np.where(adj)] -= 1

    if np.size(sz_links):
        max_sz = np.max(sz_links)
    else:
        # max_sz=0
        raise ValueError('True matrix is degenerate')
    print('max component size is %i' % max_sz)

    # estimate empirical null distribution of maximum component size by
    # generating k independent permutations
    print('estimating null distribution with %i permutations' % k)

    null = np.zeros((k,))
    hit = 0

    ind_shuff1 = np.array(range(0, y_vec.__len__()))
    ind_shuff2 = np.array(range(0, y_vec.__len__()))

    for u in range(k):
        # randomize
        np.random.shuffle(ind_shuff1)
        np.random.shuffle(ind_shuff2)
        # perform pearson corr test at each edge
        z_stat_perm = np.apply_along_axis(corr_with_vars, 1, xmat, y_vec[ind_shuff1])

        ind_r, = np.where(z_stat_perm > thresh)

        adj_perm = np.zeros((n, n))

        if extent:
            adj_perm[(ixes[0][ind_r], ixes[1][ind_r])] = 1
            adj_perm = adj_perm + adj_perm.T
        else:
            adj_perm[(ixes[0], ixes[1])] = z_stat_perm
            adj_perm = adj_perm + adj_perm.T
            adj_perm[adj_perm <= thresh] = 0

        a, sz = get_components(adj_perm)

        ind_sz, = np.where(sz > 1)
        ind_sz += 1
        nr_components_perm = np.size(ind_sz)
        sz_links_perm = np.zeros((nr_components_perm))
        for i in range(nr_components_perm):
            nodes, = np.where(ind_sz[i] == a)
            sz_links_perm[i] = np.sum(adj_perm[np.ix_(nodes, nodes)]) / 2

        if np.size(sz_links_perm):
            null[u] = np.max(sz_links_perm)
        else:
            null[u] = 0

        # compare to the true dataset
        if null[u] >= max_sz:
            hit += 1
        if verbose:
            print('permutation %i of %i.  Permutation max is %s.  Observed max'
                  ' is %s.  P-val estimate is %.3f') % (
                u, k, null[u], max_sz, hit / (u + 1))
        elif (u % (k / 10) == 0 or u == k - 1):
            print('permutation %i of %i.  p-value so far is %.3f' % (u, k,
                                                                     hit / (u + 1)))
    pvals = np.zeros((nr_components,))
    # calculate p-vals
    for i in range(nr_components):
        pvals[i] = np.size(np.where(null >= sz_links[i])) / k

    return pvals, adj, null
