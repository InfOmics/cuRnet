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
#include "Host/GraphSSSP.hpp"

using namespace graph;

GraphSSSP::GraphSSSP(const node_t _V, const edge_t _E, const EdgeType GraphDirection) :
					 GraphWeight(_V, _E, GraphDirection) {}

GraphSSSP::~GraphSSSP() {}


void GraphSSSP::ToCSR(float* weight) {
	//std::cout << "        COO To CSR...\t\t" << std::flush;

	for (int i = 0; i < COOSize; i++) {
		const node_t source = COO_Edges[i][0];
		const node_t dest = COO_Edges[i][1];
		OutDegrees[source]++;
		if (Direction == EdgeType::UNDIRECTED)
			OutDegrees[dest]++;
		else if (Direction == EdgeType::DIRECTED)
			InDegrees[dest]++;
	}

	OutNodes[0] = 0;
	std::partial_sum(OutDegrees, OutDegrees + V, OutNodes + 1);

	int* TMP = new int[V]();
	for (int i = 0; i < COOSize; i++) {
		const node_t source = COO_Edges[i][0];
		const node_t dest = COO_Edges[i][1];
        int pos = OutNodes[source] + TMP[source];
		OutEdges[pos] = dest;
        Weights[pos] = weight[i];
        TMP[source]++;
	}

	/*if (Direction == EdgeType::DIRECTED) {
		InNodes[0] = 0;
		std::partial_sum(InDegrees, InDegrees + V, InNodes + 1);

		std::fill(TMP, TMP + V, 0);
		for (int i = 0; i < COOSize; ++i) {
			const node_t dest = COO_Edges[i][1];
			InEdges[ InNodes[dest] + TMP[dest]++ ] = COO_Edges[i][0];
		}
	}
	delete[] TMP;
	std::cout << "Complete!" << std::endl << std::endl << std::flush;*/
}
