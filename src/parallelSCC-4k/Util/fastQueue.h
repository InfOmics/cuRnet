#pragma once

#include <iostream>
#include <random>
#include <chrono>
#include <assert.h>
//#define NDEBUG

namespace scc4k{

enum QPolicy {FIFO, LIFO, RAND_UNIF, RAND_EXP};

template<typename T, QPolicy POLICY = FIFO>
class FastQueue {
	private:
		int left, right, N;
		
		std::uniform_int_distribution<int> distribution;
		std::exponential_distribution<double> distributionExp;
		std::default_random_engine generator;
		
	public:
		T* Queue;
	
		FastQueue() {}
		
		FastQueue(int _N) : N(_N) {
			left = 0;
			right = 0;
			Queue = new T[N];
			
		}

		void init(int _N) {
			N = _N;
			left = 0;
			right = 0;
			Queue = new T[N];
			generator.seed( std::chrono::system_clock::now().time_since_epoch().count() );
			distributionExp.param(std::exponential_distribution<double>::param_type(3.5));
		}
		
		inline void insert(T value) {
			assert(right < N - 1);
			Queue[right++] = value;
		}
			
		inline T extract() {
			assert(left < right);
			if (POLICY == RAND_UNIF || POLICY == RAND_EXP) {
				int randomPos;
				if (POLICY == RAND_UNIF) {
					distribution.param(std::uniform_int_distribution<int>::param_type(left, right - 1));
					randomPos = distribution(generator);
					if (randomPos < left || randomPos >= right) {
						error(randomPos << "  " << left <<  "  " << right)
					}
				} else {
					double var = distributionExp(generator);
					while (var >= 1.0)
						var = distributionExp(generator);
					randomPos = left + (1 - var) * (right - left);
				}				
				assert(randomPos >= left && randomPos < right);
				T ret = Queue[randomPos];
				Queue[randomPos] = Queue[--right];
				return ret;
			}
			T ret = POLICY == FIFO ? Queue[left++] : Queue[--right];
			return ret;
		}

		inline int size() {
			return right - left;
		}

		inline T at(int i) {
			assert(i >= 0 && i < N);
			return Queue[i];
		}
		
		inline T get(int i) {
			assert(i < right - left);
			return Queue[left + i];	//Queue[right - i - 1];
		}
		
		inline void sort() {
			std::sort(Queue + left, Queue + right);
		}
		
		void print() {
			std::cout << "Queue" << std::endl;
			for (int i = left; i < right; i++)
				std::cout << Queue[i] << ' ';
			std::cout << std::endl;
		}
};

}
