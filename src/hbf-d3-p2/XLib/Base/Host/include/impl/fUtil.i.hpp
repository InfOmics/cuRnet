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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <cxxabi.h>
#include <unordered_map>

namespace fUtil {

	template<typename T>
	std::string typeStringObj(T Obj) {
		int info;
		return std::string(abi::__cxa_demangle(typeid(Obj).name(), 0, 0, &info));
	}

	template<typename T>
	std::string typeString() {
		int info;
		return std::string(abi::__cxa_demangle(typeid(T).name(), 0, 0, &info));
	}

	template<bool FAULT, typename T, typename R>
	bool Compare(T* ArrayA, R* ArrayB, const int size) {
		for (int i = 0; i < size; ++i) {
			if (ArrayA[i] != ArrayB[i]) {
				if (FAULT)
					__ERROR("Array Difference at: " << i << " -> ArrayA: " << ArrayA[i] << " ArrayB: " << ArrayB[i]);
				return false;
			}
		}
		return true;
	}

    template<bool FAULT, typename T, typename R>
    bool Compare(T* ArrayA, R* ArrayB, const int size, bool (*areEqual)(T, R)) {
        for (int i = 0; i < size; i++) {
			if (!areEqual(ArrayA[i], ArrayB[i])) {
				if (FAULT)
					__ERROR("Array Difference at: " << i << " -> ArrayA: " << ArrayA[i] << " ArrayB: " << ArrayB[i]);
				return false;
			}
		}
		return true;
	}

	template<bool FAULT, typename T, typename R>
	bool CompareAndSort(T* ArrayA, R* ArrayB, const int size) {
		T* tmpArrayA = ArrayA; R* tmpArrayB = ArrayB;
		tmpArrayA = new T[size];
		tmpArrayB = new R[size];
		std::copy(ArrayA, ArrayA + size, tmpArrayA);
		std::copy(ArrayB, ArrayB + size, tmpArrayB);
		std::sort(tmpArrayA, tmpArrayA + size);
		std::sort(tmpArrayB, tmpArrayB + size);

		bool flag = Compare<FAULT>(tmpArrayA, tmpArrayB, size);

		delete[] tmpArrayA;
		delete[] tmpArrayB;
		return flag;
	}

	template<typename T, typename R>
	R UniqueMap<T, R>::insertValue(T id) {
        static_assert(std::is_integral<R>::value, PRINT_ERR("UniqueMap accept only Integral types"));

		typename UniqueMap<T, R>::iterator IT = this->find(id);
		if (IT == this->end()) {
			R nodeID = static_cast<R>(this->size());
			this->insert(std::pair<T, R>(id, nodeID));
			return nodeID;
		}
		return IT->second;
	}
}
