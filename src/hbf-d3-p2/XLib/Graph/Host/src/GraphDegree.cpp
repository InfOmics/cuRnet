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
#include <exception>
#include "../include/GraphDegree.hpp"
#include "XLib.hpp"

namespace graph {

GraphDegree::GraphDegree(const node_t _V, const edge_t _E, const EdgeType GraphDirection) :
					       V(_V), E(_E), Direction(GraphDirection) {
	try {
		OutDegrees = new degree_t[ V ]();
	} catch(std::bad_alloc& exc) {
  		__ERROR("OUT OF MEMORY: Graph too Large !!");
	}
}

GraphDegree::~GraphDegree() {
    delete[] OutDegrees;
}

void GraphDegree::print() {
	printExt::host::printArray(OutDegrees, V, "OutDegrees\t");
}

void GraphDegree::DegreeAnalisys() {
	StreamModifier::thousandSep();
	const float avg             = (float) E / V;
	const float stdDev          = numeric::stdDeviation (OutDegrees, V, avg);
	const int zeroDegree        = std::count (OutDegrees, OutDegrees + V, 0);
	const int oneDegree         = std::count (OutDegrees, OutDegrees + V, 1);
	std::pair<int*,int*> minmax = std::minmax_element (OutDegrees, OutDegrees + V);

	std::cout << std::setprecision(1)
			  << "          Avg:  " << avg    << "\t\tOutDegree 0:  "
              << std::left << std::setw(14) << zeroDegree
              << numeric::perCent(zeroDegree, V) << " %" << std::endl
			  << "     Std. Dev:  " << stdDev << "\t\tOutDegree 1:  "
              << std::left << std::setw(14) << oneDegree
              << numeric::perCent(oneDegree, V) << " %" << std::endl
			  << "          Min:  " << *minmax.first    << "\t\t" << std::endl
			  << "          Max:  " << *minmax.second   << "\t\t" << std::endl;
	std::cout << std::endl;
	StreamModifier::resetSep();
}

} //@graph
