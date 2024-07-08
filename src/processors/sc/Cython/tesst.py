import main_cy
import MiddleSC_cy
import time

start_vanilla = time.time()
main_cy.prime_finder_vanilla(100000)
end_vanilla = time.time()

print(f"vanilla time is {end_vanilla - start_vanilla}")

start_optimized = time.time()
main_cy.prime_finder_optimized(100000)
end_optimized = time.time()

print(f"optimized time is {end_optimized - start_optimized}")