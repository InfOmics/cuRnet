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

#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace fast_queue {

    template<typename T>
	Queue<T>::Queue() {}

    template<typename T>
	Queue<T>::Queue(int _NUM) : N(_NUM) {
		left = 0;
		right = 0;
		Array = new T[N];
	}

    template<typename T>
    Queue<T>::~Queue() {
        delete[] Array;
    }

    template<typename T>
	inline void Queue<T>::init(int _NUM) {
		N = _NUM;
		left = 0;
		right = 0;
		Array = new T[N];
	}

    template<typename T>
	inline void Queue<T>::reset() {
		left = 0;
		right = 0;
	}

    template<typename T>
	inline void Queue<T>::insert(T value) {
        if (QUEUE_DEBUG && right >= N) throw std::runtime_error("Queue::insert(T) : right >= N");
		Array[right++] = value;
	}

    template<typename T>
	template<Policy POLICY>
	inline T Queue<T>::extract() {
        if (QUEUE_DEBUG && left >= right) throw std::runtime_error("Queue::extract() : left >= right");
        return POLICY == FIFO ? Array[left++] : Array[--right];
    }

    template<typename T>
    inline T Queue<T>::extract(const int i) {
    if (QUEUE_DEBUG && left + i >= right) throw std::runtime_error("Queue::extract(i) : left + i >= right");
    T ret = Array[i];
    Array[i] = Array[--right];
    return ret;
}

    template<typename T>
	inline bool Queue<T>::isEmpty() {
		return left >= right;
	}

    template<typename T>
	inline int Queue<T>::size() {
		return right - left;
	}

    template<typename T>
	inline int Queue<T>::totalSize() {
		return right;
	}

    template<typename T>
	inline T Queue<T>::at(int i) {
        if (QUEUE_DEBUG && i < 0 && i >= N) throw std::runtime_error("Queue::at(i) : i < 0 && i >= N");
		return Array[i];
	}

    template<typename T>
	inline T Queue<T>::last() {
		return Array[right - 1];
	}

    template<typename T>
	inline T Queue<T>::get(int i) {
        if (QUEUE_DEBUG && i >= right - left) throw std::runtime_error("Queue::get(i) : i >= right - left");
		return Array[left + i];	//Queue[right - i - 1];
	}

    template<typename T>
	inline void Queue<T>::sort() {
		std::sort(Array + left, Array + right);
	}

    template<typename T>
	inline void Queue<T>::print() {
		std::cout << "Queue" << std::endl;
		for (int i = left; i < right; i++)
			std::cout << Array[i] << ' ';
		std::cout << std::endl;
	}

} //@fast_queue
