"""Correctly determine the fewest number of coins to be given to a customer
such that the sum of the coins' value would equal the correct amount of change.

https://exercism.org/tracks/python/exercises/change
"""

from functools import cache
import numpy as np
import numba
import math
from time import perf_counter_ns as now

from typing import Sequence, Iterator


def main():
	test_denominations = (3, 8, 10, 25, 100)
	test_target = 189
	results: list[tuple[str,str]] = []

	for func in (change_cached, change_direct, change_simplex, change_simplex_np):
		func_name = func.__name__.removeprefix("change_")
		then = now()

		try:
			result = func(test_denominations, test_target)
		except RecursionError:
			print(f"Recursion limit reached for `{func_name}`.")
			continue
		except ValueError as e:
			if e.args[0] == "min() arg is an empty sequence":
				print(f"Failed implementation in `{func_name}`.")
				continue
			else:
				raise

		t = now() - then
		results.append((func_name + ":", f"{t:,} ns", result))

	leftw = max(len(r[0]) for r in results)
	rightw = max(len(r[1]) for r in results)
	for r in results:
		print(r[0].ljust(leftw), r[1].rjust(rightw), r[2], sep="  ")


def change_coins(change: int) -> list[int]:
	"""Is only guaranteed to work with a so called canonical coin system."""

	denominations = [1, 5, 10, 25, 100]

	coins = []
	for coin in reversed(denominations):
		count, change = divmod(change, coin)
		coins += [coin] * count

	return sorted(coins)


def change_general(denominations: Sequence[int], target: int) -> list[int]:
	"""Simple recursive implementation."""

	def impl(coins: list[int]) -> list[int]:
		current_total = sum(coins)
		if current_total == target:
			return coins

		return min((
			impl(coins + [x])
			for x in denominations
			if current_total + x <= target
			), key=len
		)

	return impl([])


def change_cached(denominations: Sequence[int], target: int) -> tuple[int,...]:
	"""Cache the results of the recursive function.
	Requires making the coin list immutable."""

	@cache
	def impl(coins: tuple[int,...]) -> tuple[int,...]:
		current_total = sum(coins)
		if current_total == target:
			return coins

		return min((
			impl(tuple(sorted(coins + (x,))))
			for x in denominations
			if current_total + x <= target
			), key=len
		)

	return impl(())


def change_cached_manual(denominations: Sequence[int], target: int):
	"""Implementing my own cache for analysis. Slower than functools.cache."""

	cache = {}

	def impl(coins: tuple[int,...]) -> tuple[int,...]:
		if coins in cache:
			return cache[coins]

		current_total = sum(coins)
		if current_total == target:
			return coins

		cache[coins] = min((
			impl(tuple(sorted(coins + (x,))))
			for x in denominations
			if current_total + x <= target
			), key=len
		)
		return cache[coins]

	impl(())
	return cache


def change_direct(denominations: Sequence[int], target: int) -> tuple[int,...]:
	"""Constructing an array where each axis represents
	a denomination, and its index the number of coins included."""

	# Limiting the size of axes if they evenly divide a larger
	# denomination. E.g. we never need to check more than 4 pennies,
	# because it's always better to just use dimes instead.
	axes = []
	for i, coin in enumerate(denominations):
		for largerCoin in denominations[i+1:]:
			div, mod = divmod(largerCoin, coin)
			if mod == 0:
				axes.append(div)
				break
		else: #nobreak
			axes.append(target // coin + 1)

	# Multiplying each index-value by its respective denomination,
	# and getting the total value for each grid cell.
	money = np.einsum("i..., i", np.indices(axes, dtype=int), denominations)

	# All indices of `money` where it equals the target value.
	matches = np.nonzero(money == target)

	# The lowest sum of indices is the the fewest coins used.
	answer = min(zip(*matches), key=sum)

	return answer


def change_simplex(denominations: Sequence[int], target: int) -> list[int] | None:
	"""Consider the array in the previous function, but traverse
	the grid cells in order of how many coins they represent.
	The first match will be the optimal answer.

	This is slower than change_direct since that uses
	numpy to speed up the bulk calculation.
	"""

	def simplexPoints(total: int, dimensions: int) -> Iterator[list[int]]:
		"""Iterates the set of all non-negative coordinates
		whose sum equals `total`. This spans a simplex surface
		in n-dimensional space, i.e. a triangle in 3D,
		a tetrahedron in 4D, etc."""

		def impl(seq: list[int]) -> Iterator[list[int]]:
			if len(seq) < dimensions - 1:
				for x in range(total - sum(seq) + 1):
					yield from impl(seq + [x])
			else:
				yield seq + [total - sum(seq)]

		return impl([])

	for coinCount in range(1, target // denominations[0] + 1):
		for coordinates in simplexPoints(coinCount, len(denominations)):
			if sum(a*b for a, b in zip(coordinates, denominations)) == target:
				return coordinates


def change_simplex_np(denominations: Sequence[int], target: int) -> np.ndarray | None:
	"""The previous function using numpy objects."""

	def simplexPoints(total: int, dimensions: int) -> np.ndarray:
		point_count = math.comb(total + dimensions - 1, dimensions - 1)
		coords = np.zeros((point_count, dimensions), dtype=int)

		simplex_loop(coords, total, point_count, dimensions)
		return coords

	denominations = np.array(denominations, dtype=int)

	for coinCount in range(1, target // denominations[0] + 1):
		coordinates = simplexPoints(coinCount, len(denominations))
		match = np.einsum("...i, i", coordinates, denominations) == target
		if np.any(match):
			return np.squeeze(coordinates[match])

_sys_int = numba.extending.as_numba_type(int) # type:ignore
_sys_np_int = np.dtype(int).name
_signature = f"void({_sys_np_int}[:,:], {_sys_int}, {_sys_int}, {_sys_int})"

@numba.njit(_signature, cache=True)
def simplex_loop(coords, total, point_count, dimensions):
	coords[0, -1] = total
	for i in range(1, point_count):
		coords[i] = coords[i - 1]

		for j in range(dimensions - 1, -1, -1):
			if coords[i, j] != 0:
				break

		coords[i, j-1] += 1
		if j == dimensions - 1:
			coords[i, j] -= 1
		else:
			coords[i, j:] = coords[0, j:]
			coords[i, -1] -= sum(coords[i, :j])


if __name__ == "__main__":
	print()
	# Despite setting eager compilation with a signature and setting cache=True,
	# numba jitted functions still require a warmup run for max speed.
	change_simplex_np((1,), 1)
	main()
