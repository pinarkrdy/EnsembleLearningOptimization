import sympy

prime_list = list()

def list_prime(number):
    prime_count = 0
    index =2
    while prime_count < number:
        while not(sympy.isprime(index)):
            index = index + 1
        prime_count = prime_count + 1
        prime_list.append(index-1)
    return prime_list

print(list_prime(10))


