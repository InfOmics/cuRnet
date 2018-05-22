/*------------------------------------------------------------------------------
Copyright © 2015 by Nicola Bombieri

H-BF is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#pragma once

namespace fast_queue {

const bool QUEUE_DEBUG = true;

enum Policy {FIFO, LIFO};

template<typename T>
class Queue {
	private:
		int left, right, N;
        T* Array;

	public:
        Queue();
		Queue(int _NUM);
        ~Queue();

		void init(int _NUM);
		inline void reset();
		inline void insert(T value);

		template<Policy POLICY = FIFO>
		inline T extract();
        inline T extract(const int i);

		inline bool isEmpty();
		inline int size();
		inline int totalSize();
		inline T at(int i);
		inline T last();
		inline T get(int i);

		inline void sort();
		void print();
};

} //@fast_queue

#include "impl/fastQueue.i.hpp"
