#pragma once

struct Branch
{
public:
	Branch() = default;
	Branch(int i, char s, double b) : index(i), sense(s), bound(b) {}
	int index = -1; /**< column/clique index */
	char sense;		/**< 'L','U','B' for lower, upper, both respectively */
	double bound;	/**< new bound */
};
