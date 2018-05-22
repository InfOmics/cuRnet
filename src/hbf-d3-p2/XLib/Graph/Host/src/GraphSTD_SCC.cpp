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
#include <iostream>
#include <map>
#include "../include/GraphSTD.hpp"

namespace graph {

const int INDEX_UNDEF = -1;

static int curr_index;
static int globalColor;

void GraphSTD::SCC_Init() {
	//InStack = Visited;
	LowLink = Distance;

	Queue.reset();
	curr_index = 0;
	globalColor = 0;
	std::fill(Index, Index + V, INDEX_UNDEF);
	std::fill(InStack.begin(), InStack.end(), false);
}

void GraphSTD::SCC(std::map<int, int>* SCC_MAP) {
	for (node_t i = 0; i < V; i++) {
		if (Index[i] == -1)
			Single_SCC(i, SCC_MAP);
	}
}

void GraphSTD::Single_SCC(const node_t source, std::map<int, int>* SCC_MAP) {
	assert(source < V);
	Queue.insert(source);

    Index[source] = curr_index;
    LowLink[source] = curr_index;
    InStack[source] = true;
    ++curr_index;

    for (edge_t i = OutNodes[source]; i < OutNodes[source + 1]; i++) {
	    const node_t dest = OutEdges[i];
	    if (Index[dest] == INDEX_UNDEF) {
        	Single_SCC(dest, SCC_MAP);
    		LowLink[source] = std::min(LowLink[source], LowLink[dest]);
	    } else if ( InStack[dest] )
		    LowLink[source] = std::min(LowLink[source], Index[dest]);
    }
    if (Index[source] == LowLink[source]) {
		int SCC_size = 0;
		node_t extracted;
		do {
			SCC_size++;
			extracted = Queue.extract<LIFO>();
			InStack[extracted] = false;
			//std::cout << extracted << ", ";
			Color[extracted] = globalColor;
		} while (extracted != source);
		//std::cout << std::endl;

		if (SCC_MAP != NULL) {
			auto it = SCC_MAP->insert(std::pair<int, int>(SCC_size, 1));
			if (!it.second)
				(*it.first).second++;
		}
    }
}

}
