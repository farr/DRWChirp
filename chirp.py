import arviz as az
from celerite2 import terms as cterms
from celerite2 import GaussianProcess as cGaussianProcess
from celerite2.jax import GaussianProcess, terms
import jax.lax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
import xarray as xr

frequency_spacing_constant = 7.72525 # This is the distance to the first sideband of the likelihood in white noise for a sinusoidal signal
frequency_derivative_spacing_constant = 15.121 # This is the distance to the first sideband of the likelihood for exp(1/2 I wdot t^2) signal

def chirp_phase(t, tc, mc):
    r"""Returns the phase of a chirp signal at time `t`, with coalescence time `tc`
    and chirp mass `mc`.

    .. math::

        \phi(t) = -\left(\\frac{t_c - t}{m_c}\\right)^{5/8}    
    """
    # Broadcast the inputs to the same shape
    t_shape = t.shape
    tc_shape = tc.shape
    mc_shape = mc.shape

    assert tc_shape == mc_shape, 'incompatible shapes in tc and mc'

    tc = tc[(slice(None),)*len(tc_shape) + (jnp.newaxis,)*len(t_shape)]
    mc = mc[(slice(None),)*len(mc_shape) + (jnp.newaxis,)*len(t_shape)]
    t = t[(jnp.newaxis,)*len(tc_shape) + (slice(None),)*len(t_shape)]

    theta = (tc - t) / mc

    return jnp.where(theta > 0, -theta**(5/8), 0)

def chirp(t, a, b, tc, mc):
    """Returns a chirping signal with `a` cosine and `b` sine quadratures,
    coalescence time `tc` and chirp mass `mc`.
    """
    phi = chirp_phase(t, tc, mc)
    phi0 = chirp_phase(jnp.array([0.0]), tc, mc)
    return a*jnp.cos(phi-phi0) + b*jnp.sin(phi-phi0)

def dummy_chirp(t, a, b, tc, mc):
    w = chirp_frequency(0.0, tc, mc)
    wdot = chirp_frequency_derivative(0.0, tc, mc)

    phase = w*t + (wdot*t)*t
    return a*jnp.cos(phase) + b*jnp.sin(phase)

def chirp_frequency(t, tc, mc):
    """Returns the instantaneous angular frequency of a chirp signal at time `t`,
    with coalescence time `tc` and chirp mass `mc`.
    """
    return jnp.where(t < tc, _chirp_frequency(t, tc, mc), 0)

def _chirp_frequency(t, tc, mc):
    theta = (tc - t) / mc

    return 5/8*theta**(-3/8)/mc

def chirp_frequency_derivative(t, tc, mc):
    """Returns the time derivative of the instantaneous angular frequency of a
    chirp signal at time `t`."""
    return jnp.where(t < tc, _chirp_frequency_derivative(t, tc, mc), 0)

def _chirp_frequency_derivative(t, tc, mc):
    theta = (tc - t)/mc

    return 15/64*theta**(-11/8)/(mc*mc)

def frequency_frequency_derivative_to_mc_tc(w, wdot):
    negative_flag = (w < 0) | (wdot < 0)

    mc = 5/8*(5/3)**(3/5)*wdot**(3/5)/w**(11/5)
    tc = 3*w/(8*wdot)

    return (jnp.where(negative_flag, 1, mc), jnp.where(negative_flag, -jnp.inf, tc))

def chirp_time(w, mc):
    """Returns the time before coalescence at which a chirp signal has frequency
    `w` with chirp mass `mc`.
    """
    theta = (8/5*mc*w)**(-8/3)
    return mc*theta

def frequency_frequencydot_bounds_to_gridsize(wbounds, wdotbounds, T):
    wgridsize = int(jnp.ceil(T*(wbounds[1] - wbounds[0]) / frequency_spacing_constant))
    wdotgridsize = int(jnp.ceil(T*T*(wdotbounds[1] - wdotbounds[0]) / frequency_derivative_spacing_constant))

    return ((wbounds[0], wgridsize), (wdotbounds[0], wdotgridsize))

def drw_chirp_loglike(t, y, yerr, mu, a, b, w, wd, drw_amp, tau):
    mc, tc = frequency_frequency_derivative_to_mc_tc(w, wd)

    chirp_signal = chirp(t, a, b, tc, mc)
    drw_process = GaussianProcess(terms.RealTerm(a=drw_amp*drw_amp, c=1/tau))
    drw_process.compute(t, yerr=yerr)
    return drw_process.log_likelihood(y-mu-chirp_signal)

def drw_chirp_model(t, y, yerr, wgrid, wdotgrid, chirp_amp_scale=None, drw_amp_scale=None, tau_min=None, tau_max=None, t_grid=None, predictive=False):
    ## TODO: think about priors.  Current prior is a bit odd (flat in broad frequency, flat in log within a frequency cell)
    tmid = jnp.median(t)

    t_centered = (t - tmid)
    T = jnp.max(t_centered) - jnp.min(t_centered)
    T2 = jnp.square(T)

    nkw = wgrid[1]
    nkwdot = wdotgrid[1]

    frequency_comb_spacing = frequency_spacing_constant / T # This is the distance to the first sideband of the likelihood in white noise for a sinusoidal signal
    wmid_restricted_unit = numpyro.sample('wmid_restricted_unit', dist.TruncatedNormal(0,1,low=0))
    wmid_restricted = numpyro.deterministic('wmid_restricted', wgrid[0] + wmid_restricted_unit * frequency_comb_spacing)

    frequency_derivative_comb_spacing = frequency_derivative_spacing_constant / T2
    wdotmid_restricted_unit = numpyro.sample('wdotmid_restricted_unit', dist.TruncatedNormal(0,1,low=0))
    wdotmid_restricted = numpyro.deterministic('wdotmid_restricted', wdotgrid[0] + wdotmid_restricted_unit * frequency_derivative_comb_spacing)

    k = numpyro.sample('k', dist.Categorical(logits=jnp.zeros(nkw*nkwdot)))
    kw = numpyro.deterministic('kw', k % nkw)
    kwdot = numpyro.deterministic('kwdot', k // nkw)

    wmid = numpyro.deterministic('wmid', wmid_restricted + frequency_comb_spacing*kw)
    wdotmid = numpyro.deterministic('wdotmid', wdotmid_restricted + frequency_derivative_comb_spacing*kwdot)

    if tau_min is None:
        tau_min = jnp.median(jnp.diff(t_centered))
    if tau_max is None:
        tau_max = jnp.max(t_centered) - jnp.min(t_centered)
    log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(tau_min), jnp.log(tau_max)))
    tau = numpyro.deterministic('tau', jnp.exp(log_tau))

    mu_unit = numpyro.sample('mu_unit', dist.Normal(0, 1))
    mu = numpyro.deterministic('mu', jnp.mean(y) + mu_unit * jnp.std(y))

    if chirp_amp_scale is None:
        chirp_amp_scale = jnp.std(y)
    a_unit = numpyro.sample('a_unit', dist.Normal(0, 1))
    b_unit = numpyro.sample('b_unit', dist.Normal(0, 1))
    a = numpyro.deterministic('a', a_unit * chirp_amp_scale)
    b = numpyro.deterministic('b', b_unit * chirp_amp_scale)
    chirp_amp = numpyro.deterministic('chirp_amp', jnp.sqrt(jnp.square(a) + jnp.square(b)))

    if drw_amp_scale is None:
        drw_amp_scale = jnp.std(y)
    log_drw_amp = numpyro.sample('log_drw_amp', dist.Normal(jnp.log(drw_amp_scale), 2))
    drw_amp = numpyro.deterministic('drw_amp', jnp.exp(log_drw_amp))

    numpyro.factor('log_like', drw_chirp_loglike(t_centered, y, yerr, mu, a, b, wmid, wdotmid, drw_amp, tau))

    if predictive:
        # log_likes has shape (2*wgridsize+1, 2*wdotgridsize+1)
        mc, tc = frequency_frequency_derivative_to_mc_tc(wmid, wdotmid)
        numpyro.deterministic('mc', mc)
        numpyro.deterministic('tc', tc + tmid)

        numpyro.deterministic('chirp_signal', chirp(t_centered, a, b, tc, mc))

        if t_grid is not None:
            t_grid_centered = t_grid - tmid
            numpyro.deterministic('chirp_signal_grid', chirp(t_grid_centered, a, b, tc, mc))

def from_numpyro_with_generated_quantities(sampler, ts, data, obs_uncert, wgrid, wdotgrid, prng_key=None, **kwargs):
    if prng_key is None:
        prng_key = jax.random.PRNGKey(np.random.randint(1<<32))

    pred = Predictive(drw_chirp_model, sampler.get_samples())(prng_key, ts, data, obs_uncert, wgrid, wdotgrid, predictive=True, **kwargs)
    trace = az.from_numpyro(sampler)

    chain_draw = ['chain', 'draw']
    shape = [trace.posterior.sizes[k] for k in chain_draw]
    coord = dict(time=ts)
    if 't_grid' in kwargs:
        coord['time_grid'] = kwargs['t_grid']

    for k, v in pred.items():
        if k not in trace.posterior and 'log_like' not in k:
            # get dimension names
            if 'chirp_signal' not in k:
                d = tuple(chain_draw)
            elif 'chirp_signal_grid' in k:
                d = tuple(chain_draw + ['time_grid'])
            else:
                d = tuple(chain_draw + ['time'])
            # get coordinates
            c = {c: coord[c] for c in d if c not in chain_draw}
            v = np.reshape(v, tuple(shape + list(v.shape[1:])))
            trace.posterior[k] = xr.DataArray(v, coords=c, dims=d)

    return trace