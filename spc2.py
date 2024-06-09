import numpy as np
from otpy.ot import Alice, Bob
from phe import paillier

# Initialize encryption with a larger key size
public_key, private_key = paillier.generate_paillier_keypair() 

# Initialization
n = 3
x = [1.0, 0.3, 0.1] # initial values for x_i
u = np.zeros(n)  # initial values for u_i
q = np.ones(n)  # q_i values
rho = 1.0
iter_max = 18

# Encrypt initial x values
x_encrypted = [public_key.encrypt(xi) for xi in x]


# Oblivious Transfer setup
secrets = [xi.ciphertext().to_bytes((xi.ciphertext().bit_length() + 7) // 8, byteorder='big') for xi in x_encrypted]
secret_length = len(secrets)

# Convert secrets to fixed-length byte arrays
#secrets = [secret.ljust(secret_length, b'\0') for secret in secrets]
print("secret length:", len(secrets))
alice = Alice(secrets, secret_length, len(secrets[0]))  # Alice sets up the OT
alice.setup()

# Bob selects which secrets he wants (for demonstration, we'll assume Bob wants all)
bob = Bob([0,1, 2])  # Selecting all indices
bob.setup()

alice.transmit()

# ADMM iterations
for k in range(iter_max):
    x_new_encrypted = []
    x_new = np.zeros(n)

    # Step 1: Update x_i using ADMM and OT
    for i in range(n):
        # Bob retrieves the required secrets
        secrets_received = bob.receive()
        x_received_encrypted = [public_key.raw_encrypt(int.from_bytes(secret, byteorder='big')) for secret in secrets_received]
        print("x received",x_received_encrypted)
        x_received = [private_key.decrypt(enc_val) for enc_val in x_received_encrypted]
        
        # Update x_i securely
        encrypted_term = public_key.encrypt(rho * (x_received[i] - u[i]))
        new_xi_encrypted = (encrypted_term / (2 * q[i] + rho)).ciphertext()
        x_new_encrypted.append(new_xi_encrypted)
        x_new[i] = private_key.decrypt(new_xi_encrypted)
    
    # Step 2: Securely update global x using homomorphic encryption
    x_global_encrypted = sum(x_new_encrypted) / n
    x_global = private_key.decrypt(x_global_encrypted)
    
    # Alice updates secrets based on new x values
    new_secrets = [xi.to_bytes((xi.bit_length() + 7) // 8, byteorder='big') for xi in x_new_encrypted]
    new_secret_length = len(new_secrets[0])
    new_secrets = [secret.ljust(new_secret_length, b'\0') for secret in new_secrets]
    new_secret_lengths = [len(secret) for secret in new_secrets]
    print(f"Iteration {k+1} - New secret lengths: {new_secret_lengths}")
    assert all(len(secret) == new_secret_length for secret in new_secrets), "All new secrets must have the same length"
    
    alice = Alice(new_secrets, new_secret_length, len(new_secrets))
    alice.transmit()
    
    # Step 3: Update dual variables u_i
    u += (x_new - x_global)
    
    # Update x for next iteration
    x = x_new
    
    print(f"Iteration {k+1}: x = {x}, x_global = {x_global}, u = {u}")

# Final result
print(f"Consensus value: {x_global}")