"""Correctly determine the fewest number of coins to be given to a customer
such that the sum of the coins' value would equal the correct amount of change.

https://exercism.org/tracks/python/exercises/change
"""

from functools import cache
import numpy as np
from typing import Sequence, Iterator


def change_coins(change: int) -> list[int]:
	"""Is only guaranteed to work with a canonical coin system."""

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

		return min(
			(impl(coins + [x])
			for x in denominations
			if current_total + x <= target)
			, key=len
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

		return min(
			(impl(tuple(sorted(coins + (x,))))
			for x in denominations
			if current_total + x <= target)
			, key=len
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

		cache[coins] = min(
			(impl(tuple(sorted(coins + (x,))))
			for x in denominations
			if current_total + x <= target)
			, key=len
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

	# def coinSum(*indices):
	# 	return sum(a*b for a, b in zip(indices, denominations))
	# money = np.fromfunction(coinSum, axes)

	# The same as above but slightly faster.
	money = np.einsum("i..., i", np.indices(axes, dtype=int), denominations)

	return min(zip(*np.nonzero(money == target)), key=sum)


def change_simplex(denominations: Sequence[int], target: int) -> list[int] | None:
	"""Consider the array in the previous function, but traverse
	its elements in order of how many coins they represent.
	The first match will be the optimal answer."""

	rank = len(denominations)

	def simplexPoints(coinCount: int) -> Iterator[list[int]]:
		"""Iterates the set of all non-negative coordinates
		whose sum equals coinCount. This spans a simplex surface
		in [rank]-dimensional space."""

		def impl(seq: list[int]):
			if len(seq) < rank - 1:
				for x in range(coinCount - sum(seq) + 1):
					yield from impl(seq + [x])
			else:
				yield seq + [coinCount - sum(seq)]
		return impl([])

	for coinCount in range(1, target // denominations[0] + 1):
		for coordinates in simplexPoints(coinCount):
			if sum(a*b for a, b in zip(coordinates, denominations)) == target:
				return coordinates


from time import perf_counter as now

for func in (change_cached, change_direct, change_simplex):
	try:
		then = now()
		func((1,5,10,25,100), 189)
		t = now() - then
		print(f"{t:.4e} seconds")

	except RecursionError:
		print("Too big")
