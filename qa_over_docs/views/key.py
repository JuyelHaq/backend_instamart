from cryptography.fernet import Fernet

# Generate a key for encryption
key = Fernet.generate_key()
print(key)
cipher_suite = Fernet(key)

# Assuming user token is stored in the variable `usertoken`
#usertoken = cursor.fetchone()[0]

# Encrypt the user token
#encrypted_token = cipher_suite.encrypt(usertoken.encode())

#print("Encrypted Token:", encrypted_token)

