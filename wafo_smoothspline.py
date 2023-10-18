import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg


# stuff needed from the polynomial library of wafo
def polyint(p, m=1, k=None):
    """
    Return an antiderivative (indefinite integral) of a polynomial.

    The returned order `m` antiderivative `P` of polynomial `p` satisfies
    :math:`\\frac{d^m}{dx^m}P(x) = p(x)` and is defined up to `m - 1`
    integration constants `k`. The constants determine the low-order
    polynomial part

    .. math:: \\frac{k_{m-1}}{0!} x^0 + \\ldots + \\frac{k_0}{(m-1)!}x^{m-1}

    of `P` so that :math:`P^{(j)}(0) = k_{m-j-1}`.

    Parameters
    ----------
    p : {array_like, np.poly1d}
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `np.poly1d`.
    m : int, optional
        Order of the antiderivative. (Default: 1)
    k : {None, list of `m` scalars, scalar}, optional
        Integration constants. They are given in the order of integration:
        those corresponding to highest-order terms come first.

        If ``None`` (default), all constants are assumed to be zero.
        If `m = 1`, a single scalar can be given instead of a list.

    See Also
    --------
    polyder : derivative of a polynomial
    np.poly1d.integ : equivalent method

    Examples
    --------
    The defining property of the antiderivative:

    >>> p = np.poly1d([1,1,1])
    >>> P = np.polyint(p)
    >>> P
    np.poly1d([ 0.33333333,  0.5       ,  1.        ,  0.        ])
    >>> np.polyder(P) == p
    True

    The integration constants default to zero, but can be specified:

    >>> P = np.polyint(p, 3)
    >>> P(0)
    0.0
    >>> np.polyder(P)(0)
    0.0
    >>> np.polyder(P, 2)(0)
    0.0
    >>> P = np.polyint(p, 3, k=[6, 5, 3])
    >>> P.coefficients.tolist()
    [0.016666666666666666, 0.041666666666666664, 0.16666666666666666, 3.0,
    5.0, 3.0]

    Note that 3 = 6 / 2!, and that the constants are given in the order of
    integrations. Constant of the highest-order polynomial term comes first:

    >>> np.polyder(P, 2)(0)
    6.0
    >>> np.polyder(P, 1)(0)
    5.0
    >>> P(0)
    3.0

    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    if k is None:
        k = np.zeros(m, float)
    k = np.atleast_1d(k)
    if len(k) == 1 and m > 1:
        k = k[0] * np.ones(m, float)
    if len(k) < m:
        raise ValueError(
            "k must be a scalar or a rank-1 array of length 1 or >m.")
    truepoly = isinstance(p, np.poly1d)
    p = np.asarray(p)
    if m == 0:
        if truepoly:
            return np.poly1d(p)
        return p
    else:
        ix = np.arange(len(p), 0, -1)
        if p.ndim > 1:
            ix = ix[..., newaxis]
            pieces = p.shape[-1]
            k0 = k[0] * np.ones((1, pieces), dtype=int)
        else:
            k0 = [k[0]]
        y = np.concatenate((p.__truediv__(ix), k0), axis=0)

        val = polyint(y, m - 1, k=k[1:])
        if truepoly:
            return np.poly1d(val)
        return val


def polyder(p, m=1):
    """
    Return the derivative of the specified order of a polynomial.

    Parameters
    ----------
    p : np.poly1d or sequence
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `np.poly1d`.
    m : int, optional
        Order of differentiation (default: 1)

    Returns
    -------
    der : np.poly1d
        A new polynomial representing the derivative.

    See Also
    --------
    polyint : Anti-derivative of a polynomial.
    np.poly1d : Class for one-dimensional polynomials.

    Examples
    --------
    The derivative of the polynomial :math:`x^3 + x^2 + x^1 + 1` is:

    >>> p = np.poly1d([1,1,1,1])
    >>> p2 = np.polyder(p)
    >>> p2
    np.poly1d([3, 2, 1])

    which evaluates to:

    >>> p2(2.)
    17.0

    We can verify this, approximating the derivative with
    ``(f(x + h) - f(x))/h``:

    >>> (p(2. + 0.001) - p(2.)) / 0.001
    17.007000999997857

    The fourth-order derivative of a 3rd-order polynomial is zero:

    >>> np.polyder(p, 2)
    np.poly1d([6, 2])
    >>> np.polyder(p, 3)
    np.poly1d([6])
    >>> np.polyder(p, 4)
    np.poly1d([ 0.])

    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")
    truepoly = isinstance(p, np.poly1d)
    p = np.asarray(p)
    if m == 0:
        if truepoly:
            return np.poly1d(p)
        return p
    else:
        n = len(p) - 1
        ix = np.arange(n, 0, -1)
        if p.ndim > 1:
            ix = ix[..., newaxis]
        y = ix * p[:-1]
        val = polyder(y, m - 1)
        if truepoly:
            return np.poly1d(val)
        return val


def polyreloc(p, x, y=0.0):
    """
    Relocate polynomial

    The polynomial `p` is relocated by "moving" it `x`
    units along the x-axis and `y` units along the y-axis.
    So the polynomial `r` is relative to the point (x,y) as
    the polynomial `p` is relative to the point (0,0).

    Parameters
    ----------
    p : array-like, np.poly1d
        vector or matrix of column vectors of polynomial coefficients to
        relocate. (Polynomial coefficients are in decreasing order.)
    x : scalar
        distance to relocate P along x-axis
    y : scalar
        distance to relocate P along y-axis (default 0)

    Returns
    -------
    r : ndarray, np.poly1d
        vector/matrix/np.poly1d of relocated polynomial coefficients.

    See also
    --------
    polyrescl

    Example
    -------
    >>> import numpy as np
    >>> p = np.arange(6); p.shape = (2,-1)
    >>> np.polyval(p,0)
    array([3, 4, 5])
    >>> np.polyval(p,1)
    array([3, 5, 7])
    >>> r = polyreloc(p,-1) # move to the left along x-axis
    >>> np.polyval(r,-1)    # = polyval(p,0)
    array([3, 4, 5])
    >>> np.polyval(r,0)     # = polyval(p,1)
    array([3, 5, 7])
    """

    truepoly = isinstance(p, np.poly1d)
    r = np.atleast_1d(p).copy()
    n = r.shape[0]

    # Relocate polynomial using Horner's algorithm
    for ii in range(n, 1, -1):
        for i in range(1, ii):
            r[i] = r[i] - x * r[i - 1]
    r[-1] = r[-1] + y
    if r.ndim > 1 and r.shape[-1] == 1:
        r.shape = (r.size, )
    if truepoly:
        r = np.poly1d(r)
    return r


class PPform(object):
    """The ppform of the piecewise polynomials
                    is given in terms of coefficients and breaks.
    The polynomial in the ith interval is
        x_{i} <= x < x_{i+1}

    S_i = sum(coefs[m,i]*(x-breaks[i])^(k-m), m=0..k)
    where k is the degree of the polynomial.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> coef = np.array([[1,1]]) # unit step function
    >>> coef = np.array([[1,1],[0,1]]) # linear from 0 to 2
    >>> coef = np.array([[1,1],[1,1],[0,2]]) # linear from 0 to 2
    >>> breaks = [0,1,2]
    >>> self = PPform(coef, breaks)
    >>> x = linspace(-1,3)
    >>> h=plt.plot(x,self(x))
    """

    def __init__(self, coeffs, breaks, fill=0.0, sort=False, a=None, b=None):
        if sort:
            self.breaks = np.sort(breaks)
        else:
            self.breaks = np.asarray(breaks)
        if a is None:
            a = self.breaks[0]
        if b is None:
            b = self.breaks[-1]
        self.coeffs = np.asarray(coeffs)
        self.order = self.coeffs.shape[0]
        self.fill = fill
        self.a = a
        self.b = b

    def __call__(self, xnew):
        saveshape = np.shape(xnew)
        xnew = np.ravel(xnew)
        res = np.empty_like(xnew)
        mask = (self.a <= xnew) & (xnew <= self.b)
        res[~mask] = self.fill
        xx = xnew.compress(mask)
        indxs = np.searchsorted(self.breaks[:-1], xx) - 1
        indxs = indxs.clip(0, len(self.breaks))
        pp = self.coeffs
        dx = xx - self.breaks.take(indxs)

        v = pp[0, indxs]
        for i in range(1, self.order):
            v = dx * v + pp[i, indxs]
        values = v

        res[mask] = values
        res.shape = saveshape
        return res

    def linear_extrapolate(self, output=True):
        '''
        Return 1D PPform which extrapolate linearly outside its basic interval
        '''

        max_order = 2

        if self.order <= max_order:
            if output:
                return self
            else:
                return
        breaks = self.breaks.copy()
        coefs = self.coeffs.copy()
        # pieces = len(breaks) - 1

        # Add new breaks beyond each end
        breaks2add = breaks[[0, -1]] + np.array([-1, 1])
        newbreaks = np.hstack([breaks2add[0], breaks, breaks2add[1]])

        dx = newbreaks[[0, -2]] - breaks[[0, -2]]

        dx = dx.ravel()

        # Get coefficients for the new last polynomial piece (a_n)
        # by just relocate the previous last polynomial and
        # then set all terms of order > maxOrder to zero

        a_nn = coefs[:, -1]
        dxN = dx[-1]

        a_n = polyreloc(a_nn, -dxN)  # Relocate last polynomial
        # set to zero all terms of order > maxOrder
        a_n[0:self.order - max_order] = 0

        # Get the coefficients for the new first piece (a_1)
        # by first setting all terms of order > maxOrder to zero and then
        # relocate the polynomial.

        # Set to zero all terms of order > maxOrder, i.e., not using them
        a_11 = coefs[self.order - max_order::, 0]
        dx1 = dx[0]

        a_1 = polyreloc(a_11, -dx1)  # Relocate first polynomial
        a_1 = np.hstack([np.zeros(self.order - max_order), a_1])

        newcoefs = np.hstack([a_1.reshape(-1, 1), coefs, a_n.reshape(-1, 1)])
        if output:
            return PPform(newcoefs, newbreaks, a=-np.inf, b=np.inf)
        else:
            self.coeffs = newcoefs
            self.breaks = newbreaks
            self.a = -np.inf
            self.b = np.inf

    def derivative(self):
        """
        Return first derivative of the piecewise polynomial
        """

        cof = polyder(self.coeffs)
        brks = self.breaks.copy()
        return PPform(cof, brks, fill=self.fill)

    def integrate(self):
        """
        Return the indefinite integral of the piecewise polynomial
        """
        cof = polyint(self.coeffs)

        pieces = len(self.breaks) - 1
        if 1 < pieces:
            # evaluate each integrated polynomial at the right endpoint of its
            # interval
            xs = np.diff(self.breaks[:-1, ...], axis=0)
            index = np.arange(pieces - 1)

            vv = xs * cof[0, index]
            k = self.order
            for i in range(1, k):
                vv = xs * (vv + cof[i, index])

            cof[-1] = np.hstack((0, vv)).cumsum()

        return PPform(cof, self.breaks, fill=self.fill)


class SmoothSpline(PPform):
    """
    Cubic Smoothing Spline.

    Parameters
    ----------
    x : array-like
        x-coordinates of data. (vector)
    y : array-like
        y-coordinates of data. (vector or matrix)
    p : real scalar
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    lin_extrap : bool
        if False regular smoothing spline
        if True a smoothing spline with a constraint on the ends to
        ensure linear extrapolation outside the range of the data (default)
    var : array-like
        variance of each y(i) (default  1)

    Returns
    -------
    pp : ppform
        If xx is not given, return self-form of the spline.

    Given the approximate values

        y(i) = g(x(i))+e(i)

    of some smooth function, g, where e(i) is the error. SMOOTH tries to
    recover g from y by constructing a function, f, which  minimizes

      p * sum (Y(i) - f(X(i)))^2/d2(i)  +  (1-p) * int (f'')^2


    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0,1)
    >>> y = np.exp(x)+1e-1*np.random.randn(x.size)
    >>> pp9 = SmoothSpline(x, y, p=.9)
    >>> pp99 = SmoothSpline(x, y, p=.99, var=0.01)
    >>> h=plt.plot(x,y, x,pp99(x),'g', x,pp9(x),'k', x,np.exp(x),'r')

    See also
    --------
    lc2tr, dat2tr


    References
    ----------
    Carl de Boor (1978)
    'Practical Guide to Splines'
    Springer Verlag
    Uses EqXIV.6--9, self 239
    """

    def __init__(self, xx, yy, p=None, lin_extrap=True, var=1):
        coefs, brks = self._compute_coefs(xx, yy, p, var)
        super(SmoothSpline, self).__init__(coefs, brks)
        if lin_extrap:
            self.linear_extrapolate(output=False)

    def _compute_coefs(self, xx, yy, p=None, var=1):
        x, y = np.atleast_1d(xx, yy)
        x = x.ravel()
        dx = np.diff(x)
        must_sort = (dx < 0).any()
        if must_sort:
            ind = x.argsort()
            x = x[ind]
            y = y[..., ind]
            dx = np.diff(x)

        n = len(x)

        # ndy = y.ndim
        szy = y.shape

        nd = np.prod(szy[:-1])
        ny = szy[-1]

        if n < 2:
            raise ValueError('There must be >=2 data points.')
        elif (dx <= 0).any():
            raise ValueError('Two consecutive values in x can not be equal.')
        elif n != ny:
            raise ValueError('x and y must have the same length.')

        dydx = np.diff(y) / dx

        if (n == 2):  # % straight line
            coefs = np.vstack([dydx.ravel(), y[0, :]])
        else:

            dx1 = 1. / dx
            D = sparse.spdiags(var * np.ones(n), 0, n, n)  # The variance

            u, p = self._compute_u(p, D, dydx, dx, dx1, n)
            dx1.shape = (n - 1, -1)
            dx.shape = (n - 1, -1)
            zrs = np.zeros(int(nd))
            if p < 1:
                # faster than yi-6*(1-p)*Q*u
                Qu = D * np.diff(np.vstack([
                    zrs,
                    np.diff(np.vstack([zrs, u, zrs]), axis=0) * dx1, zrs
                ]),
                                 axis=0)
                ai = (y - (6 * (1 - p) * Qu).T).T
            else:
                ai = y.reshape(n, -1)

            # The piecewise polynominals are written as
            # fi=ai+bi*(x-xi)+ci*(x-xi)^2+di*(x-xi)^3
            # where the derivatives in the knots according to Carl de Boor are:
            #    ddfi  = 6*p*[0;u] = 2*ci;
            #    dddfi = 2*diff([ci;0])./dx = 6*di;
            #    dfi   = diff(ai)./dx-(ci+di.*dx).*dx = bi;

            ci = np.vstack([zrs, 3 * p * u])
            di = (np.diff(np.vstack([ci, zrs]), axis=0) * dx1 / 3)
            bi = (np.diff(ai, axis=0) * dx1 - (ci + di * dx) * dx)
            ai = ai[:n - 1, ...]
            if nd > 1:
                di = di.T
                ci = ci.T
                ai = ai.T
            if not any(di):
                if not any(ci):
                    coefs = np.vstack([bi.ravel(), ai.ravel()])
                else:
                    coefs = np.vstack([ci.ravel(), bi.ravel(), ai.ravel()])
            else:
                coefs = np.vstack(
                    [di.ravel(),
                     ci.ravel(),
                     bi.ravel(),
                     ai.ravel()])

        return coefs, x

    def _compute_u(self, p, D, dydx, dx, dx1, n):
        if p is None or p != 0:
            data = [dx[1:n - 1], 2 * (dx[:n - 2] + dx[1:n - 1]), dx[:n - 2]]
            R = sparse.spdiags(data, [-1, 0, 1], n - 2, n - 2)

        if p is None or p < 1:
            Q = sparse.spdiags(
                [dx1[:n - 2], -(dx1[:n - 2] + dx1[1:n - 1]), dx1[1:n - 1]],
                [0, -1, -2], n, n - 2)
            QDQ = (Q.T * D * Q)
            if p is None or p < 0:
                # Estimate p
                p = 1. / \
                    (1. + QDQ.diagonal().sum() /
                     (100. * R.diagonal().sum() ** 2))

            if p == 0:
                QQ = 6 * QDQ
            else:
                QQ = (6 * (1 - p)) * (QDQ) + p * R
        else:
            QQ = R

        # Make sure it uses symmetric matrix solver
        ddydx = np.diff(dydx, axis=0)
        u = 2 * scipy.sparse.linalg.spsolve((QQ + QQ.T), ddydx)
        return u.reshape(n - 2, -1), p
