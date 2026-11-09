import jax

print(f"Available devices: {jax.devices()}")
print(f"Available local devices: {jax.local_devices()}")
print(f"Default Backend: {jax.default_backend()}")
