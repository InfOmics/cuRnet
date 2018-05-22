#pragma once

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include <iostream>
#include <string>
#include <sstream>
#include <numeric>
#include <cmath>
#include <iomanip>      	// setprecision
#include <cstdlib>			// exit
#include <locale>			// numpunct<char>
#include <ostream>			// color
#include <algorithm>		// sorting

#define INFOErr		{std::cerr << " Error in " << __FILE__ << " : " << __func__ << " (line: " << __LINE__ << ")" << endl << endl;}
#define error(a)	{std::cerr << "\n ! ERROR : " << a << std::endl << std::endl; exit(EXIT_FAILURE);}
#define error2(a)	{std::cerr << "\n ! ERROR : " << a << std::endl << std::endl; INFOErr exit(EXIT_FAILURE);}

#define DIV(a,b) (((a) + (b) - 1)/(b))		// q = ceil(x / y)   =>   q = (x + y - 1) / y;

#define	def_SWITCH(x, forward, visitType, dynPar)	switch (x) {			\
							case -1:			\
							case 0:	fun(1, forward, visitType, dynPar)		\
								break;			\
							case 1: fun(2, forward, visitType, dynPar)		\
								break;			\
							case 2: fun(4, forward, visitType, dynPar)		\
								break;			\
							case 3: fun(8, forward, visitType, dynPar)		\
								break;			\
							case 4: fun(16, forward, visitType, dynPar)		\
								break;			\
							case 5: fun(32, forward, visitType, dynPar)		\
								break;			\
							case 6: fun(64, forward, visitType, dynPar)		\
								break;			\
							case 7: fun(128, forward, visitType, dynPar)	\
								break;			\
							default: fun(32, forward, visitType, dynPar)	\
						}

#define	def_SWITCHB(x, forward, visitType, dynPar)	switch (x) {			\
							case -1:			\
							case 0:	funB(1, forward, visitType, dynPar)		\
								break;			\
							case 1: funB(2, forward, visitType, dynPar)		\
								break;			\
							case 2: funB(4, forward, visitType, dynPar)		\
								break;			\
							case 3: funB(8, forward, visitType, dynPar)		\
								break;			\
							case 4: funB(16, forward, visitType, dynPar)	\
								break;			\
							case 5: funB(32, forward, visitType, dynPar)	\
								break;			\
							case 6: funB(64, forward, visitType, dynPar)	\
								break;			\
							case 7: funB(128, forward, visitType, dynPar)	\
								break;			\
							default: fun(32, forward, visitType, dynPar)	\
						}


#if __cplusplus < 199711L && ! __GXX_EXPERIMENTAL_CXX0X__
namespace std {

	template <class ForwardIterator>
	pair<ForwardIterator,ForwardIterator>  minmax_element (ForwardIterator first, ForwardIterator last) {
		pair<ForwardIterator,ForwardIterator> minmax;
		minmax.first = first;
		minmax.second = first;
		while (first != last) {
			if (*first < minmax.first)
				minmax.first = *first;
			else if (*first > minmax.second)
				minmax.second = *first;
			first++;
		}
		return minmax;
	}

	template <class RandomAccessIterator, class RandomNumberGenerator>
	void random_shuffle (RandomAccessIterator first, RandomAccessIterator last, RandomNumberGenerator& gen) {
		iterator_traits<RandomAccessIterator>::difference_type i, n;
		n = (last-first);
		for (i=n-1; i>0; --i) {
			swap (first[i],first[gen(i+1)]);
		}
	}
}
#endif

namespace scc4k{

namespace mt {
	//#if __cplusplus < 199711L
		#define __STDC_LIMIT_MACROS
		#include <stdint.h>

		#undef RAND_MAX
//		#include "mersenne-twister.h"
	//#endif
}

// ----------- META-PROGRAMMING -------------------

template <int _N, int CURRENT_VAL = _N, int COUNTL = 0>
struct _Log2 {
	enum { VALUE = _Log2< _N, (CURRENT_VAL >> 1), COUNTL + 1>::VALUE };
};

template <int _N, int COUNTL>
struct _Log2<_N, 0, COUNTL> {
	enum { VALUE = (1 << (COUNTL - 1) < _N) ? COUNTL : COUNTL - 1 };
};


template <int POW, int COUNTL = POW>
struct _Pow2 {
	enum { VALUE = POW == 0 ? 1 : _Pow2< (POW << 1), COUNTL - 1 >::VALUE };
};

template <int POW>
struct _Pow2<POW, 0> {
	enum { VALUE = POW };
};

template <int MOD>
struct _Mod2 {
	enum { VALUE = MOD - 1 };
};

template <int NUM>
struct _Mask2 {
	enum { VALUE = ~((1 << _Log2<NUM>::VALUE) - 1) };
};

// NEAREST POW 2

template <int NUM, int C>
struct _OR_SHIFT {
	enum { VALUE = _OR_SHIFT<NUM | (NUM >> C), C * 2>::VALUE};
};

template <int NUM>
struct _OR_SHIFT<NUM, 32> {
	enum { VALUE = NUM + 1 };
};

template <int NUM>
struct _NearestPow2 {
	enum { VALUE = _OR_SHIFT<NUM - 1, 1>::VALUE };
};
	/*v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;*/

namespace Color {
    enum Code {
        FG_RED       = 31, FG_GREEN     = 32, FG_YELLOW    = 33,
        FG_BLUE      = 34, FG_MAGENTA   = 35, FG_CYAN      = 36,
		FG_L_GRAY    = 37, FG_D_GREY    = 90, FG_L_RED     = 91,
        FG_L_GREEN   = 92, FG_L_YELLOW  = 93, FG_L_BLUE    = 94,
		FG_L_MAGENTA = 95, FG_L_CYAN    = 96, FG_WHITE	 = 97,
        FG_DEFAULT  = 39,
    };

	std::ostream& operator<<(std::ostream& os, const Code& mod);
}

namespace SetFormat {
    enum CodeS {
        SET_BOLD      = 1, SET_DIM       = 2,
		SET_UNDERLINE = 4, SET_RESET     = 0,
    };

	std::ostream& operator<<(std::ostream& os, const CodeS& mod);
}

namespace fUtil {

	void memInfoPrint(size_t total, size_t free, size_t Req);

	void memInfoHost(int Req);

	struct myseps : std::numpunct<char> {
		// use space as separator
		char do_thousands_sep() const { return ','; }

		// digits are grouped by 3 digits each
		std::string do_grouping() const { return "\3"; }
	};

	std::string extractFileName(std::string s);

	template<bool SORT = false, bool FAULT = true, typename T>
	bool Compare(T* ArrayA, T* ArrayB, const int size);

	// --------------------------- MATH ---------------------------------------------------

	unsigned nearestPower2(unsigned v);
	unsigned log_2(unsigned n);
	bool isPositiveInteger(const std::string& s);
	float perCent(int part, int max);

	template<class T>
    float average(T* Input, const int size);

	template<class T>
	float stdDeviation(T* Input, const int size);

	// --------------------------- IMPLEMENTATION ---------------------------------------------------

	template<class T>
    float average(T* Input, const int size) {
		const T sum = std::accumulate(Input, Input + size, 0);
		return (float) sum / size;
	}

	template<class T>
	float stdDeviation(T* Input, const int size) {
        const float avg = averageB(Input, size);
		return stdDeviation(Input, size, avg);
	}

	template<class T>
	float stdDeviation(T* Input, const int size, float avg) {
		float sum = 0;
		for (int i = 0; i < size; ++i)
			sum += std::pow(Input[i] - avg, 2);
		return std::sqrt(sum / size);
	}

	template<bool SORT, bool FAULT, typename T, typename R>
	bool Compare(T* ArrayA, R* ArrayB, const int size) {
		T* tmpArrayA = ArrayA; R* tmpArrayB = ArrayB;
		if (SORT) {
			tmpArrayA = new T[size];
			tmpArrayB = new R[size];
			std::copy(ArrayA, ArrayA + size, tmpArrayA);
			std::copy(ArrayB, ArrayB + size, tmpArrayB);
			std::sort(tmpArrayA, tmpArrayA + size);
			std::sort(tmpArrayB, tmpArrayB + size);
		}
		for (int i = 0; i < size; ++i) {
			if (tmpArrayA[i] != tmpArrayB[i]) {
				if (FAULT)
					error(" Array Difference at: " << i << " -> ArrayA: " << tmpArrayA[i] << " ArrayB: " << tmpArrayB[i]);
				if (SORT) {
					delete[] tmpArrayA;
					delete[] tmpArrayB;
				}
				return false;
			}
		}
		if (SORT) {
			delete[] tmpArrayA;
			delete[] tmpArrayB;
		}
		return true;
	}
}

}
