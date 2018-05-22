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
#include <algorithm>
#include <exception>
#include <vector_types.h>	//int2
#include <vector_functions.hpp>

#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "../include/GraphSTD.hpp"
#include "XLib.hpp"

namespace graph {

GraphSTD::GraphSTD(const node_t _V, const edge_t _E, const EdgeType _edgeType) :
					GraphDegree(_V, _E, _edgeType) {
	try {
		OutNodes = new edge_t[ V + 1 ];
		OutEdges = new node_t[ E ];
		COO_Edges = new node_t2[ E ];
		if (_edgeType == EdgeType::UNDIRECTED) {
            InDegrees = OutDegrees;
            return;
        }
		InNodes = new edge_t[ V + 1 ];
		InEdges = new node_t[ E ];
		InDegrees = new degree_t[ V ]();
	}
	catch(std::bad_alloc& exc) {
  		__ERROR("OUT OF MEMORY: Graph too Large !!");
	}
}

GraphSTD::~GraphSTD() {
    delete[] OutNodes;
	delete[] OutEdges;
	delete[] COO_Edges;
	if (Direction == EdgeType::UNDIRECTED)
		return;
	delete[] InNodes;
	delete[] InEdges;
	delete[] InDegrees;
}

void GraphSTD::ToCSR() {
	std::cout << "        COO To CSR...\t\t" << std::flush;

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
		OutEdges[ OutNodes[source] + TMP[source]++ ] = dest;
		if (Direction == EdgeType::UNDIRECTED)
			OutEdges[ OutNodes[dest] + TMP[dest]++ ] = source;
	}

	if (Direction == EdgeType::DIRECTED) {
		InNodes[0] = 0;
		std::partial_sum(InDegrees, InDegrees + V, InNodes + 1);

		std::fill(TMP, TMP + V, 0);
		for (int i = 0; i < COOSize; ++i) {
			const node_t dest = COO_Edges[i][1];
			InEdges[ InNodes[dest] + TMP[dest]++ ] = COO_Edges[i][0];
		}
	}
	delete[] TMP;
	std::cout << "Complete!" << std::endl << std::endl << std::flush;
}

void GraphSTD::toBinary(const char* File) {
	const int fd = open(File, O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600);

	const int fileSize = ((V + 1) * 2 + V * 2 + E * 2 + 3) * sizeof(int);
	std::cout << "Graph To binary file: " << File << " (" << (fileSize / (1 << 20)) << ") MB" << std::endl;

	lseek(fd, fileSize - 1, SEEK_SET);
	write(fd, "", 1);

	int* memory_mapped = (int*) mmap(NULL, fileSize, PROT_WRITE, MAP_SHARED, fd, 0);
	if (memory_mapped == MAP_FAILED)
		__ERROR("memory_mapped error");
	madvise(memory_mapped, fileSize, MADV_SEQUENTIAL);
	fUtil::Progress progress(fileSize);

	memory_mapped[0] = V;
	memory_mapped[1] = E;
	memory_mapped[2] = static_cast<int>(Direction);

	memory_mapped += 3;
	std::copy(OutNodes, OutNodes + V + 1, memory_mapped);
	progress.perCent(V + 1);

	memory_mapped += V + 1;
	std::copy(InNodes, InNodes + V + 1, memory_mapped);
	progress.perCent((V + 1) * 2);

	memory_mapped += V + 1;
	std::copy(OutDegrees, OutDegrees + V, memory_mapped);
	progress.perCent((V + 1) * 2 + V);

	memory_mapped += V;
	std::copy(InDegrees, InDegrees + V, memory_mapped);
	progress.perCent((V + 1) * 2 + V * 2);

	memory_mapped += V;
	std::copy(OutEdges, OutEdges + E, memory_mapped);
	progress.perCent((V + 1) * 2 + V * 2 + E);

	memory_mapped += E;
	std::copy(InEdges, InEdges + E, memory_mapped);

	munmap(memory_mapped, fileSize);
	close(fd);
}

void GraphSTD::print() {
	printExt::host::printArray(OutNodes, V + 1, "OutNodes\t");
	printExt::host::printArray(OutEdges, E, "OutEdges\t");
	printExt::host::printArray(OutDegrees, V, "OutDegrees\t");
	if (Direction == EdgeType::UNDIRECTED)
		return;
	printExt::host::printArray(InNodes, V + 1, "InNodes\t\t");
	printExt::host::printArray(InEdges, E, "InEdges\t\t");
	printExt::host::printArray(InDegrees, V, "InDegrees\t");
}


void GraphSTD::DegreeAnalisys() {
	StreamModifier::thousandSep();
	const float avg             = (float) E / V;
	const float stdDev          = numeric::stdDeviation (OutDegrees, V, avg);
	const int zeroDegree        = std::count (OutDegrees, OutDegrees + V, 0);
	const int oneDegree         = std::count (OutDegrees, OutDegrees + V, 1);
	std::pair<int*,int*> minmax = std::minmax_element (OutDegrees, OutDegrees + V);

	std::cout << std::setprecision(1)
			  << "          Avg:  " << avg    << "\t\tOutDegree 0:  " << std::left << std::setw(14) << zeroDegree << numeric::perCent(zeroDegree, V) << " %" << std::endl
			  << "     Std. Dev:  " << stdDev << "\t\tOutDegree 1:  " << std::left << std::setw(14) << oneDegree << numeric::perCent(oneDegree, V) << " %" << std::endl
			  << "          Min:  " << *minmax.first    << "\t\t" << std::endl
			  << "          Max:  " << *minmax.second   << "\t\t" << std::endl;
	if (Direction == EdgeType::DIRECTED)
		std::cout << "\t\t\t\t InDegrees 0:  " << std::count (InDegrees, InDegrees + V, 0) << std::endl
				  << "\t\t\t\t InDegrees 1:  " << std::count (InDegrees, InDegrees + V, 1) << std::endl;
	std::cout << std::endl;
	StreamModifier::resetSep();
}

} //@graph
