import numpy as np
from otpy import Alice, Bob
from phe import paillier


# Initialize variables
n = 3
x0 = np.array([1.0, 0.3, 0.1])
q = np.array([1.0, 1.0, 1.0])
rho = 1.0
iter_max = 18

# Initialize Paillier encryption
public_key, private_key = paillier.generate_paillier_keypair()

# Function to perform the ADMM update
def admm_update(x, u, x_global, i):
    return (rho * (x_global - u[i])) / (q[i] + rho)

# ADMM iterations
x = x0.copy()
u = np.zeros_like(x0)

# Initialize OT senders (Alice) and receivers (Bob) for each node
alices = [Alice() for _ in range(n)]
bobs = [Bob() for _ in range(n)]

for k in range(iter_max):
    # Step 1: Update x_i securely
    for i in range(n):
        x_new = admm_update(x, u, np.mean(x + u), i)
        x[i] = x_new
        
        # Securely share x_new with other nodes using OT
        for j in range(n):
            if i != j:
                # Alice prepares the choices (the encrypted values)
                alice = alices[i]
                encrypted_value_0 = public_key.encrypt(x_new)
                encrypted_value_1 = public_key.encrypt(x_new)
                alice.prepare_choices([encrypted_value_0, encrypted_value_1])

                # Bob chooses and receives the encrypted value
                bob = bobs[j]
                bob.set_choices([0, 1])  # Setting the choices, assuming choice 0 for simplicity
                encrypted_value = bob.get_result(alice.get_sent_data())
                decrypted_value = private_key.decrypt(encrypted_value)
                
                # Assign the decrypted value to x[j] if it's not already set
                if x[j] != decrypted_value:
                    x[j] = decrypted_value

    # Step 2: Update global x
    x_global = np.mean(x)
    
    # Step 3: Update u_i securely
    for i in range(n):
        u[i] = u[i] + x[i] - x_global
        
        # Securely share u_i with other nodes using OT
        for j in range(n):
            if i != j:
                # Alice prepares the choices (the encrypted values)
                alice = alices[i]
                encrypted_value_0 = public_key.encrypt(u[i])
                encrypted_value_1 = public_key.encrypt(u[i])
                alice.prepare_choices([encrypted_value_0, encrypted_value_1])

                # Bob chooses and receives the encrypted value
                bob = bobs[j]
                bob.set_choices([0, 1])  # Setting the choices, assuming choice 0 for simplicity
                encrypted_value = bob.get_result(alice.get_sent_data())
                decrypted_value = private_key.decrypt(encrypted_value)
                
                # Assign the decrypted value to u[j] if it's not already set
                if u[j] != decrypted_value:
                    u[j] = decrypted_value

print("Final consensus value:", x_global)