import jax.numpy as jnp

def chirp_phase(t, tc, mc):
    r"""Returns the phase of a chirp signal at time `t`, with coalescence time `tc`
    and chirp mass `mc`.

    .. math::

        \phi(t) = -\left(\\frac{t_c - t}{m_c}\\right)^{5/8}    
    """
    return jnp.where(t < tc, _chirp_phase(t, tc, mc), 0)

def _chirp_phase(t, tc, mc):
    theta = (tc - t) / mc

    return -theta**(5/8)

def chirp(t, a, b, tc, mc):
    """Returns a chirping signal with `a` cosine and `b` sine quadratures,
    coalescence time `tc` and chirp mass `mc`.
    """
    phi = chirp_phase(t, tc, mc)
    phi0 = chirp_phase(0, tc, mc)
    return a*jnp.cos(phi-phi0) + b*jnp.sin(phi-phi0)

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
    mc = 5/8*(5/3)**(3/5)*wdot**(3/5)/w**(11/5)
    tc = 3*w/(8*wdot)

    return (mc, tc)

def chirp_time(w, mc):
    """Returns the time before coalescence at which a chirp signal has frequency
    `w` with chirp mass `mc`.
    """
    theta = (8/5*mc*w)**(-8/3)
    return mc*theta