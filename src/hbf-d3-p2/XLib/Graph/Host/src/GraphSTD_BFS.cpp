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
#include "../include/GraphSTD.hpp"

namespace graph {

void GraphSTD::BFS_Init() {
    Queue = new fast_queue::Queue<node_t>(V);
	Visited.resize(V);
	Distance = new dist_t[ V ];
    BFS_Reset();
}

void GraphSTD::BFS_Reset() {
	Queue->reset();

	std::fill(Visited.begin(), Visited.end(), false);
	std::fill(Distance, Distance + V, std::numeric_limits<dist_t>::max());
}

void GraphSTD::BFS(const node_t source) {
	Visited[source] = true;
	Distance[source] = 0;
	Queue->insert(source);

	while (!Queue->isEmpty()) {
		const node_t next = Queue->extract();

		for (edge_t i = OutNodes[next]; i < OutNodes[next + 1]; i++) {
			const node_t dest = OutEdges[i];

			if (!Visited[dest]) {
				Visited[dest] = true;
				Distance[dest] = Distance[next] + 1;
				Queue->insert(dest);
			}
		}
	}
}

int GraphSTD::BFS_visitedNodes() {
	return Queue->totalSize();
}

int GraphSTD::BFS_visitedEdges() {
	if (Queue->totalSize() == V)
		return E;
	int sum = 0;
	for (int i = 0; i < Queue->totalSize(); ++i)
		sum += OutDegrees[ Queue->at(i) ];
	return sum;
}

int GraphSTD::BFS_getEccentricity() {
	return Distance[Queue->last()];
}

void GraphSTD::BFS_Frontier(std::vector<node_t>& Frontiers) {
	dist_t oldDistance = 0;
	Frontiers.resize(0);
	Frontiers.push_back(1);
	//int edges = 0;
	//int level = 0;

	while (!Queue->isEmpty()) {
		const node_t qNode = Queue->extract();

		if (Distance[qNode] > oldDistance) {
			Frontiers.push_back(Queue->size() + 1);
			oldDistance = Distance[qNode];
			//std::cout << std::endl << std::endl << "LEVEL " << oldDistance + 1 <<  " Qsize " << Queue->size() + 1 << " Edges " << edges << std::endl;
		//	edges = 0;
		}
		//edges += OutDegree[qNode];
		//std::cout <<  qNode << ' ';

		for (edge_t i = OutNodes[qNode]; i < OutNodes[qNode + 1]; ++i) {
			const node_t dest = OutEdges[i];

			//std::cout << dest << " ";
			if (!Visited[dest]) {
				Visited[dest] = true;
				Distance[dest] = Distance[qNode] + 1;
				Queue->insert(dest);
			}
		}
	}
}

void GraphSTD::TwoSweepDiameter() {
	/*int* Distance = new int[N];

	int lower_bound = 0, upper_bound;
	for (int i = 0; i < N; i++) {
		BFS_Init(Distance);
		BFS(rand_source);
		BFS_Init(Queue->last(), Distance);
		if (BFS_getEccentricity() > lower_bound)
			lower_bound = BFS_getEccentricity();

		if (UNDIRECTED) {
			BFS_Init(Distance);
			BFS(highDegree_source);
			if (BFS_getEccentricity() > lower_bound)
				lower_bound = BFS_getEccentricity();

			BFS_Init( Distance );
			BFS( Queue->last() );
			if (BFS_getEccentricity() < upper_bound)
				upper_bound = BFS_getEccentricity();

			if (lower_bound >= upper_bound)
				break;
		}
	}
	std::cout << lower_bound << std::endl;*/
}

} //@graph
