import jax.numpy as jnp

def chirp_phase(t, tc, mc):
    return jnp.where(t < tc, _chirp_phase(t, tc, mc), 0)

def _chirp_phase(t, tc, mc):
    theta = (tc - t) / mc

    return -theta**(5/8)

def chirp(t, a, b, tc, mc):
    phi = chirp_phase(t, tc, mc)
    return a*jnp.cos(phi) + b*jnp.sin(phi)

def chirp_frequency(t, tc, mc):
    return jnp.where(t < tc, _chirp_frequency(t, tc, mc), 0)

def _chirp_frequency(t, tc, mc):
    theta = (tc - t) / mc

    return 5/8*theta**(-3/8)/mc