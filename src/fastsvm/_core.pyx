# cython: language_level=3
"""Private, validated Cython/C numerical kernels. Not a stable public API.

All low-level entry points validate shapes/indices before unchecked loops.
The optimization algorithms are implemented here, not delegated to sklearn.
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport exp, pow, tanh, fabs, sqrt, INFINITY, isfinite
from libc.stdint cimport uint64_t, int64_t
from cpython.exc cimport PyErr_CheckSignals
from cython.parallel cimport prange

cnp.import_array()

cdef extern from *:
    """
    static int fastsvm_has_openmp(void) {
    #ifdef _OPENMP
        return 1;
    #else
        return 0;
    #endif
    }
    """
    int fastsvm_has_openmp() noexcept nogil


def build_info():
    return {"openmp": bool(fastsvm_has_openmp()), "backend": "Cython/C", "precision": "float64"}

cdef inline double dot(const double* x, const double* z, Py_ssize_t d) noexcept nogil:
    # Independent accumulators shorten the floating-point dependency chain.
    cdef double a=0, b=0, c=0, e=0
    cdef Py_ssize_t j=0
    while j+3 < d:
        a += x[j]*z[j]
        b += x[j+1]*z[j+1]
        c += x[j+2]*z[j+2]
        e += x[j+3]*z[j+3]
        j += 4
    while j < d:
        a += x[j]*z[j]
        j += 1
    return (a+b)+(c+e)

cdef inline double kernel(const double* x, const double* z, Py_ssize_t d,
                          int kind, double gamma, double coef0, int degree) noexcept nogil:
    cdef Py_ssize_t j
    cdef double value=0, diff, b=0, c=0, e=0, d0, d1, d2, d3
    if gamma==0:
        if kind==1 or kind==4:
            return 1.
        if kind==2:
            return pow(coef0,degree)
        if kind==3:
            return tanh(coef0)
    if kind == 1:  # Direct distances avoid catastrophic cancellation for large offsets.
        j=0
        while j+3<d:
            d0=x[j]-z[j]
            d1=x[j+1]-z[j+1]
            d2=x[j+2]-z[j+2]
            d3=x[j+3]-z[j+3]
            value+=d0*d0
            b+=d1*d1
            c+=d2*d2
            e+=d3*d3
            j+=4
        while j<d:
            diff=x[j]-z[j]
            value+=diff*diff
            j+=1
        return exp(-gamma*((value+b)+(c+e)))
    if kind == 4:  # Laplacian L1 kernel.
        for j in range(d):
            value += fabs(x[j]-z[j])
        return exp(-gamma*value)
    value = dot(x,z,d)
    if kind == 0:
        return value
    if kind == 2:
        return pow(gamma*value+coef0, degree)
    return tanh(gamma*value+coef0)


def kernel_matrix(const double[:, ::1] X, const double[:, ::1] Z,
                  int kind, double gamma, double coef0, int degree, int n_threads=1):
    cdef Py_ssize_t i,j,n=X.shape[0],m=Z.shape[0],d=X.shape[1]
    if d == 0 or Z.shape[1] != d or n_threads < 1 or kind < 0 or kind > 4:
        raise ValueError("Invalid kernel arguments or incompatible feature counts.")
    cdef double[:, ::1] out = np.empty((n,m),dtype=np.float64)
    for i in prange(n, nogil=True, num_threads=n_threads, schedule='static',
                    use_threads_if=n_threads>1 and n*m*d>=100000):
        for j in range(m):
            out[i,j]=kernel(&X[i,0],&Z[j,0],d,kind,gamma,coef0,degree)
    if not np.isfinite(np.asarray(out)).all():
        raise ValueError("Non-finite kernel values; rescale data or kernel parameters.")
    return np.asarray(out)


cdef double predict_row(const double* x, const double[:, ::1] sv,
                        const double[::1] coeff, int kind, double gamma,
                        double coef0, int degree, double bias) noexcept nogil:
    cdef Py_ssize_t j
    cdef double value=bias
    for j in range(sv.shape[0]):
        value += coeff[j]*kernel(x,&sv[j,0],sv.shape[1],kind,gamma,coef0,degree)
    return value


def kernel_predict(const double[:, ::1] X, const double[:, ::1] sv,
                   const double[::1] coeff, int kind, double gamma,
                   double coef0, int degree, double bias, int n_threads=1):
    cdef Py_ssize_t i,n=X.shape[0]
    if X.shape[1] == 0 or X.shape[1] != sv.shape[1] or sv.shape[0] != coeff.shape[0]:
        raise ValueError("Invalid support vectors or coefficients.")
    if n_threads < 1 or kind < 0 or kind > 4:
        raise ValueError("Invalid kernel arguments.")
    cdef double[::1] out=np.empty(n)
    for i in prange(n, nogil=True, num_threads=n_threads, schedule='static',
                    use_threads_if=n_threads>1 and n*sv.shape[0]*X.shape[1]>=100000):
        out[i]=predict_row(&X[i,0],sv,coeff,kind,gamma,coef0,degree,bias)
    if not np.isfinite(np.asarray(out)).all():
        raise ValueError("Non-finite decision values; rescale inputs.")
    return np.asarray(out)


cdef class RowCache:
    """O(1) LRU lookup/eviction; stores raw K rows once even for SVR's 2n dual."""
    cdef const double[:, ::1] X
    cdef double[:, ::1] rows
    cdef int64_t[::1] location, owner, prev, nxt
    cdef Py_ssize_t head, tail, used, capacity, n, d
    cdef int kind, degree
    cdef double gamma, coef0
    cdef long hits, misses
    cdef bint invalid

    def __init__(self, const double[:, ::1] X, int kind, double gamma,
                 double coef0, int degree, double cache_mb):
        self.X=X
        self.n=X.shape[0]
        self.d=X.shape[1]
        self.kind=kind
        self.gamma=gamma
        self.coef0=coef0
        self.degree=degree
        self.capacity=<Py_ssize_t>min(<double>self.n, max(2.,cache_mb/(8.0*self.n/1048576.)))
        self.rows=np.empty((self.capacity,self.n),dtype=np.float64)
        self.location=np.full(self.n,-1,dtype=np.int64)
        self.owner=np.full(self.capacity,-1,dtype=np.int64)
        self.prev=np.full(self.capacity,-1,dtype=np.int64)
        self.nxt=np.full(self.capacity,-1,dtype=np.int64)
        self.head=-1
        self.tail=-1
        self.used=0
        self.hits=0
        self.misses=0
        self.invalid=False

    cdef void touch(self, Py_ssize_t slot) noexcept nogil:
        cdef Py_ssize_t p,q
        if self.head==slot:
            return
        p=self.prev[slot]
        q=self.nxt[slot]
        if p>=0:
            self.nxt[p]=q
        if q>=0:
            self.prev[q]=p
        if self.tail==slot:
            self.tail=p
        self.prev[slot]=-1
        self.nxt[slot]=self.head
        if self.head>=0:
            self.prev[self.head]=slot
        self.head=slot
        if self.tail<0:
            self.tail=slot

    cdef double* row(self, Py_ssize_t i) noexcept nogil:
        cdef Py_ssize_t slot=self.location[i],j
        cdef double v
        if slot>=0:
            self.hits+=1
            self.touch(slot)
            return &self.rows[slot,0]
        self.misses+=1
        if self.used < self.capacity:
            slot=self.used
            self.used+=1
        else:
            slot=self.tail
            self.location[self.owner[slot]]=-1
        self.owner[slot]=i
        self.location[i]=slot
        for j in range(self.n):
            if self.kind==5:
                v=self.X[i,j]
            else:
                v=kernel(&self.X[i,0],&self.X[j,0],self.d,self.kind,
                         self.gamma,self.coef0,self.degree)
            self.rows[slot,j]=v
            if not isfinite(v):
                self.invalid=True
        self.touch(slot)
        return &self.rows[slot,0]

cdef inline bint up(double a,double c,double s) noexcept nogil:
    return c>0 and ((s>0 and a<c) or (s<0 and a>0))

cdef inline bint low(double a,double c,double s) noexcept nogil:
    return c>0 and ((s>0 and a>0) or (s<0 and a<c))


def smo(const double[:, ::1] X, const int64_t[::1] mapping,
        const double[::1] signs, const double[::1] p, const double[::1] bounds,
        const double[::1] initial, int kind, double gamma, double coef0,
        int degree, double tol, long max_iter, double cache_mb,
        bint shrinking=True, bint history=False):
    """Minimize .5*a'Q*a+p'a with box constraints and preserved s'a.

    Q_ij=s_i*s_j*K(mapping_i,mapping_j). Second-order working-set selection.
    Input feasibility is checked by the public estimators; here initial box
    feasibility and all shapes/indices are independently checked.
    """
    cdef Py_ssize_t n=mapping.shape[0], i,j,k,mi,mj,free_count,kk,active_n,min_active,kept
    cdef long it=0, shrink_steps=0
    cdef double shrink_threshold
    cdef double vmax,vmin,v,eta,gain,best,t,cap_i,cap_j,bias,free_sum,violation
    cdef double norm2,dual,ai,aj,dai,daj
    cdef double* ri
    cdef double* rj
    if n==0 or X.shape[0]==0 or X.shape[1]==0:
        raise ValueError("Empty optimization problem.")
    if (signs.shape[0] != n or p.shape[0] != n or
            bounds.shape[0] != n or initial.shape[0] != n):
        raise ValueError("Incompatible optimization shapes.")
    if (np.asarray(mapping)<0).any() or (np.asarray(mapping)>=X.shape[0]).any():
        raise ValueError("Invalid dual-to-sample mapping.")
    if kind<0 or kind>5 or (kind==5 and X.shape[0]!=X.shape[1]):
        raise ValueError("Invalid kernel or Gram matrix shape.")
    if tol<=0 or max_iter<=0 or cache_mb<=0:
        raise ValueError("Invalid solver controls.")
    if not np.isfinite(np.asarray(X)).all():
        raise ValueError("Optimization arrays must be finite.")
    if not np.isfinite(np.asarray(signs)).all():
        raise ValueError("Optimization arrays must be finite.")
    if not np.isfinite(np.asarray(p)).all():
        raise ValueError("Optimization arrays must be finite.")
    if not np.isfinite(np.asarray(bounds)).all():
        raise ValueError("Optimization arrays must be finite.")
    if not np.isfinite(np.asarray(initial)).all():
        raise ValueError("Optimization arrays must be finite.")
    if not np.isin(np.asarray(signs),[-1,1]).all():
        raise ValueError("Signs must be +/-1.")
    if (np.asarray(bounds)<0).any() or (np.asarray(initial)<0).any() or (np.asarray(initial)>np.asarray(bounds)).any():
        raise ValueError("Infeasible initial dual variables.")
    cdef double[::1] a=np.array(initial,copy=True)
    cdef double[::1] G=np.array(p,copy=True)
    cdef double[::1] diag=np.empty(n)
    cdef int64_t[::1] active=np.arange(n,dtype=np.int64)
    active_n=n
    min_active=n
    cdef RowCache cache=RowCache(X,kind,gamma,coef0,degree,cache_mb)
    cdef list trace=[]
    with nogil:
        for k in range(n):
            mi=mapping[k]
            if kind==5:
                diag[k]=X[mi,mi]
            else:
                diag[k]=kernel(&X[mi,0],&X[mi,0],X.shape[1],kind,gamma,coef0,degree)
            if a[k]!=0:
                ri=cache.row(mi)
                for j in range(n):
                    G[j]+=signs[j]*signs[k]*a[k]*ri[mapping[j]]
        while it < max_iter:
            vmax=-INFINITY
            vmin=INFINITY
            i=-1
            for kk in range(active_n):
                k=active[kk]
                v=-signs[k]*G[k]
                if up(a[k],bounds[k],signs[k]) and v>vmax:
                    vmax=v
                    i=k
                if low(a[k],bounds[k],signs[k]) and v<vmin:
                    vmin=v
            violation=max(0.,vmax-vmin)
            if cache.invalid:
                break
            if i<0 or not isfinite(vmin) or violation<=tol:
                if active_n<n:
                    # Gradients were kept up to date even for inactive variables.
                    # Never accept active-set convergence without a full recheck.
                    active_n=n
                    for kk in range(n):
                        active[kk]=kk
                    continue
                break
            ri=cache.row(mapping[i])
            best=-1
            j=-1
            for kk in range(active_n):
                k=active[kk]
                if low(a[k],bounds[k],signs[k]):
                    v=vmax+signs[k]*G[k]
                    if v>0:
                        eta=diag[i]+diag[k]-2*ri[mapping[k]]
                        if eta<=0:
                            eta=1e-12
                        gain=v*v/eta
                        if gain>best:
                            best=gain
                            j=k
            if j<0:
                break
            mj=mapping[j]
            eta=diag[i]+diag[j]-2*ri[mj]
            if eta<=0:
                eta=1e-12
            t=(vmax+signs[j]*G[j])/eta
            cap_i=bounds[i]-a[i] if signs[i]>0 else a[i]
            cap_j=a[j] if signs[j]>0 else bounds[j]-a[j]
            t=min(t,min(cap_i,cap_j))
            # Snap limiting variables to their exact bounds. Tiny round-off in
            # the other variable is retained in the actual gradient update.
            ai=a[i]
            aj=a[j]
            a[i]=min(bounds[i],max(0.,ai+signs[i]*t))
            a[j]=min(bounds[j],max(0.,aj-signs[j]*t))
            if t==cap_i:
                a[i]=bounds[i] if signs[i]>0 else 0.
            if t==cap_j:
                a[j]=0. if signs[j]>0 else bounds[j]
            dai=(a[i]-ai)*signs[i]
            daj=(a[j]-aj)*signs[j]
            rj=cache.row(mj)  # >=2 slots ensures the just-touched i row survives.
            for k in range(n):
                G[k]+=signs[k]*(dai*ri[mapping[k]]+daj*rj[mapping[k]])
            it+=1
            if shrinking and it%100==0 and active_n>2:
                # Conservative heuristic: remove bound variables whose only
                # feasible direction is well outside the current KKT interval.
                # No dual variable is discarded; final full reactivation is mandatory.
                shrink_threshold=max(10*tol,0.1*violation)
                kept=0
                for kk in range(active_n):
                    k=active[kk]
                    v=-signs[k]*G[k]
                    if bounds[k]==0:
                        continue
                    if up(a[k],bounds[k],signs[k]) and not low(a[k],bounds[k],signs[k]) and v<vmin-shrink_threshold:
                        continue
                    if low(a[k],bounds[k],signs[k]) and not up(a[k],bounds[k],signs[k]) and v>vmax+shrink_threshold:
                        continue
                    active[kept]=k
                    kept+=1
                active_n=kept
                min_active=min(min_active,active_n)
                shrink_steps+=1
            if shrinking and it%2000==0:
                active_n=n
                for kk in range(n):
                    active[kk]=kk
            if it % 1024 == 0:
                with gil:
                    PyErr_CheckSignals()
            if history and (it==1 or it%100==0):
                dual=0
                for k in range(n):
                    dual-=0.5*a[k]*(G[k]+p[k])
                with gil:
                    trace.append((it,dual,violation))
        # Always re-scan, including max_iter exits: report the FINAL residual.
        vmax=-INFINITY
        vmin=INFINITY
        free_sum=0
        free_count=0
        norm2=0
        for k in range(n):
            v=-signs[k]*G[k]
            if up(a[k],bounds[k],signs[k]):
                vmax=max(vmax,v)
            if low(a[k],bounds[k],signs[k]):
                vmin=min(vmin,v)
            if a[k]>0 and a[k]<bounds[k]:
                free_sum+=v
                free_count+=1
            norm2+=a[k]*(G[k]-p[k])
        if free_count>0:
            bias=free_sum/free_count
        elif isfinite(vmax) and isfinite(vmin):
            bias=(vmax+vmin)*0.5
        elif isfinite(vmax):
            bias=vmax
        elif isfinite(vmin):
            bias=vmin
        else:
            bias=0
        violation=max(0.,vmax-vmin)
    if cache.invalid or not np.isfinite(np.asarray(G)).all() or not isfinite(norm2):
        raise ValueError("Kernel optimization overflowed; rescale data/kernel parameters.")
    if history:
        trace.append((it,float(-0.5*np.dot(np.asarray(a),np.asarray(G)+np.asarray(p))),violation))
    return {"alpha": np.asarray(a), "gradient": np.asarray(G), "intercept": bias,
            "n_iter": it, "kkt_violation": violation, "converged": violation<=tol,
            "norm_squared": norm2, "history": np.asarray(trace).reshape(-1,3),
            "cache_hits": cache.hits, "cache_misses": cache.misses,
            "cache_bytes": cache.capacity*cache.n*8,
            "min_active_size": min_active, "shrinking_steps": shrink_steps}


cdef inline uint64_t rng_next(uint64_t* state) noexcept nogil:
    cdef uint64_t x=state[0]
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    state[0]=x
    return x * <uint64_t>2685821657736338717

cdef inline double row_dot(const double[:,::1] X, const double[::1] data,
                          const int64_t[::1] indices, const int64_t[::1] indptr,
                          const double[::1] w, Py_ssize_t i, bint sparse) noexcept nogil:
    cdef Py_ssize_t k
    cdef double v=0
    if sparse:
        for k in range(indptr[i],indptr[i+1]):
            v+=data[k]*w[indices[k]]
        return v
    return dot(&X[i,0],&w[0],X.shape[1])

cdef inline void row_add(const double[:,::1] X, const double[::1] data,
                         const int64_t[::1] indices, const int64_t[::1] indptr,
                         double[::1] w, Py_ssize_t i, double delta,
                         bint sparse) noexcept nogil:
    cdef Py_ssize_t k
    if sparse:
        for k in range(indptr[i],indptr[i+1]):
            w[indices[k]]+=delta*data[k]
    else:
        for k in range(X.shape[1]):
            w[k]+=delta*X[i,k]

cdef inline double pg_value(double a,double g,double bound,double eps,
                            bint regression) noexcept nogil:
    if regression:
        if a>0:
            g+=eps
        elif a<0:
            g-=eps
        else:
            return max(0.,fabs(g)-eps)
        if a<=-bound:
            return min(g,0.)
        if a>=bound:
            return max(g,0.)
        return g
    if a<=0:
        return min(g,0.)
    if a>=bound:
        return max(g,0.)
    return g


def linear_cd(const double[:,::1] X, const double[::1] data,
              const int64_t[::1] indices, const int64_t[::1] indptr,
              Py_ssize_t n_features, const double[::1] target,
              const double[::1] cost, int mode, double epsilon, double bias_scale,
              double tol, long max_iter, uint64_t seed,
              bint shrinking=True, bint history=False):
    """Randomized dual CD: modes 0/1 hinge/squared hinge; 2/3 eps/Squared eps.

    The intercept is a regularized synthetic feature, matching LIBLINEAR's
    convention. No shared RNG and no shared optimizer state.
    """
    cdef Py_ssize_t n=target.shape[0],d=n_features,i,j,k,t,active_n
    cdef long epoch=0
    cdef bint sparse=indptr.shape[0]>0, regression=mode>=2, squared=mode%2==1
    cdef double z,g,pg,a_new,delta,sign,bw=0,pgmax=INFINITY,q,v,dual
    cdef uint64_t state=seed+<uint64_t>1
    if state==0:
        state=1
    if n==0 or d==0 or cost.shape[0]!=n or mode<0 or mode>3:
        raise ValueError("Invalid linear optimization problem.")
    if tol<=0 or max_iter<=0 or epsilon<0 or bias_scale<0:
        raise ValueError("Invalid linear solver controls.")
    if sparse:
        if indptr.shape[0]!=n+1 or data.shape[0]!=indices.shape[0] or indptr[0]!=0 or indptr[n]!=data.shape[0]:
            raise ValueError("Invalid CSR shape.")
        if (np.diff(np.asarray(indptr))<0).any() or (np.asarray(indices)<0).any() or (np.asarray(indices)>=d).any():
            raise ValueError("Invalid CSR indices.")
    elif X.shape[0]!=n or X.shape[1]!=d:
        raise ValueError("Invalid dense matrix shape.")
    if not np.isfinite(np.asarray(X)).all():
        raise ValueError("Linear solver arrays must be finite.")
    if not np.isfinite(np.asarray(data)).all():
        raise ValueError("Linear solver arrays must be finite.")
    if not np.isfinite(np.asarray(target)).all():
        raise ValueError("Linear solver arrays must be finite.")
    if not np.isfinite(np.asarray(cost)).all():
        raise ValueError("Linear solver arrays must be finite.")
    if (np.asarray(cost)<0).any() or not (np.asarray(cost)>0).any():
        raise ValueError("Costs must be nonnegative with a positive entry.")
    if not regression and not np.isin(np.asarray(target),[-1,1]).all():
        raise ValueError("Classification target must be +/-1.")
    cdef double[::1] a=np.zeros(n),w=np.zeros(d),diag=np.zeros(n),Q=np.zeros(n),upper=np.zeros(n)
    cdef int64_t[::1] active=np.arange(n,dtype=np.int64)
    cdef list trace=[]
    active_n=n
    with nogil:
        for i in range(n):
            if cost[i]<=0:
                continue
            diag[i]=1./(2*cost[i]) if squared else 0.
            upper[i]=INFINITY if squared else cost[i]
            q=bias_scale*bias_scale+diag[i]
            if sparse:
                for k in range(indptr[i],indptr[i+1]):
                    q+=data[k]*data[k]
            else:
                q+=dot(&X[i,0],&X[i,0],d)
            Q[i]=q
        while epoch < max_iter:
            if shrinking and epoch%10==0:
                active_n=n
                for i in range(n):
                    active[i]=i
            for k in range(active_n-1,0,-1):
                j=<Py_ssize_t>(rng_next(&state)%<uint64_t>(k+1))
                t=active[k]
                active[k]=active[j]
                active[j]=t
            k=0
            pgmax=0
            while k<active_n:
                i=active[k]
                if cost[i]<=0:
                    k+=1
                    continue
                sign=1. if regression else target[i]
                z=row_dot(X,data,indices,indptr,w,i,sparse)+bias_scale*bw
                g=z-target[i]+diag[i]*a[i] if regression else sign*z-1+diag[i]*a[i]
                pg=pg_value(a[i],g,upper[i],epsilon,regression)
                pgmax=max(pgmax,fabs(pg))
                if shrinking and pg==0 and (
                    (not regression and ((a[i]==0 and g>10*tol) or (a[i]==upper[i] and g<-10*tol))) or
                    (regression and ((a[i]==0 and fabs(g)<epsilon-10*tol) or
                     (a[i]==-upper[i] and g-epsilon>10*tol) or
                     (a[i]==upper[i] and g+epsilon<-10*tol)))):
                    active_n-=1
                    t=active[k]
                    active[k]=active[active_n]
                    active[active_n]=t
                    continue
                if fabs(pg)>1e-15:
                    if Q[i]>0:
                        z=a[i]-g/Q[i]
                        if regression:
                            v=epsilon/Q[i]
                            if z>v:
                                z-=v
                            elif z<-v:
                                z+=v
                            else:
                                z=0
                            a_new=min(upper[i],max(-upper[i],z))
                        else:
                            a_new=min(upper[i],max(0.,z))
                    elif regression:
                        a_new=upper[i] if target[i]>epsilon else (-upper[i] if target[i]<-epsilon else 0.)
                    else:
                        a_new=upper[i]
                    delta=(a_new-a[i])*sign
                    a[i]=a_new
                    row_add(X,data,indices,indptr,w,i,delta,sparse)
                    bw+=delta*bias_scale
                k+=1
            epoch+=1
            if history:
                dual=-0.5*(dot(&w[0],&w[0],d)+bw*bw)
                for i in range(n):
                    dual+=(target[i]*a[i]-epsilon*fabs(a[i]) if regression else a[i])-0.5*diag[i]*a[i]*a[i]
                with gil:
                    trace.append((epoch,dual,pgmax))
            if epoch%20==0:
                with gil:
                    PyErr_CheckSignals()
            if pgmax<=tol:
                # Full, freshly evaluated PG norm is the convergence certificate.
                pgmax=0
                for i in range(n):
                    if cost[i]<=0:
                        continue
                    z=row_dot(X,data,indices,indptr,w,i,sparse)+bias_scale*bw
                    g=z-target[i]+diag[i]*a[i] if regression else target[i]*z-1+diag[i]*a[i]
                    pgmax=max(pgmax,fabs(pg_value(a[i],g,upper[i],epsilon,regression)))
                if pgmax<=tol:
                    break
                active_n=n
                for i in range(n):
                    active[i]=i
        pgmax=0
        for i in range(n):
            if cost[i]<=0:
                continue
            z=row_dot(X,data,indices,indptr,w,i,sparse)+bias_scale*bw
            g=z-target[i]+diag[i]*a[i] if regression else target[i]*z-1+diag[i]*a[i]
            pgmax=max(pgmax,fabs(pg_value(a[i],g,upper[i],epsilon,regression)))
    if not np.isfinite(np.asarray(w)).all() or not isfinite(bw) or not isfinite(pgmax):
        raise ValueError("Linear optimization overflowed; standardize the inputs.")
    if history:
        trace.append((epoch,trace[len(trace)-1][1] if trace else 0.,pgmax))
    return {"alpha":np.asarray(a),"coef":np.asarray(w),"intercept":bw*bias_scale,
            "bias_weight":bw,"n_iter":epoch,"kkt_violation":pgmax,"converged":pgmax<=tol,
            "history":np.asarray(trace).reshape(-1,3)}
