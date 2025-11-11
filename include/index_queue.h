#pragma once

#include <vector>
#include "tool_assert.h"

/** Circular queue to store up to k indices of type Integer in the range [0,n) */
template <typename Integer>
class IndexQueue
{
public:
	IndexQueue(Integer _k, Integer _n) : k{_k}, n{_n}, indices(k), ismember(n, false)
	{
		FP_ASSERT(k > 0);
		FP_ASSERT(n > 0);
	}
	bool has(Integer x) const
	{
		FP_ASSERT(0 <= x && x < n);
		return ismember[x];
	}
	bool empty() const { return (count == 0); }
	bool full() const { return (count == k); }
	size_t size() const
	{
		return (size_t)(count);
	}
	void clear()
	{
		if (count)
		{
			ismember.clear();
			count = 0;
			first = 0;
			last = 0;
		}
		else
		{
			FP_ASSERT(first == last);
		}
	}
	Integer operator[](Integer itr) const
	{
		FP_ASSERT((0 <= itr) && (itr < count));
		return indices[(first + itr) % k];
	}
	void push(Integer x)
	{
		FP_ASSERT(0 <= x && x < n);
		// do nothing if the index is already in the queue
		if (has(x))
			return;
		// if full pop the oldest element (this is basically overwriting old data)
		if (full())
			pop();
		// add x
		FP_ASSERT(!full());
		ismember[x] = true;
		indices[last] = x;
		increment(last);
		count++;
		FP_ASSERT((0 < count) && (count <= k));
		FP_ASSERT((0 <= first) && (first < k));
		FP_ASSERT((0 <= last) && (last < k));
	}
	Integer pop()
	{
		FP_ASSERT(!empty());
		Integer ret = indices[first];
		ismember[ret] = false;
		increment(first);
		count--;
		FP_ASSERT((0 <= count) && (count < k));
		FP_ASSERT((0 <= first) && (first < k));
		FP_ASSERT((0 <= last) && (last < k));
		return ret;
	}

private:
	Integer k;
	Integer first = 0;
	Integer last = 0;
	Integer count = 0;
	Integer n;
	std::vector<Integer> indices;
	std::vector<bool> ismember;
	void increment(Integer &itr) const
	{
		itr++;
		if (itr == k)
			itr = 0;
	}
};
