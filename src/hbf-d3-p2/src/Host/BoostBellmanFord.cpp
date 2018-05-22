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
#include <boost/config.hpp>
#include <vector>
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>
#include "Host/GraphSSSP.hpp"
#include "XLib.hpp"
using namespace timer;

struct EdgeProperties {
	int weight;
};

void GraphSSSP::BoostBellmanFord(const node_t source) {
	using namespace boost;

	typedef std::pair < int, int > Edge;
	typedef adjacency_list < vecS, vecS, directedS, no_property, EdgeProperties> Graph;

	Edge* edgesB = new Edge[E];
	int k = 0;
	for (int i = 0; i < V; i++) {
		for (int j = OutNodes[i]; j < OutNodes[i+1]; j++)
			edgesB[k++] = Edge(i, OutEdges[j]);
	}
	Graph g(edgesB, edgesB + E, V);

	graph_traits < Graph >::edge_iterator ei, ei_end;
	property_map<Graph, int EdgeProperties::*>::type weight_pmap = get(&EdgeProperties::weight, g);
	int i = 0;
	for (boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei, ++i)
		weight_pmap[*ei] = Weights[i];

	std::vector<int> distance(V, (std::numeric_limits < int >::max)());
	std::vector<std::size_t> parent(V);
	for (i = 0; i < V; ++i)
		parent[i] = i;
	distance[source] = 0;

	Timer<HOST> TM;
    TM.start();

	bellman_ford_shortest_paths(g, int (V), weight_map(weight_pmap).distance_map(&distance[0]).predecessor_map(&parent[0]));

	TM.getTime("Boost BellmanFord");

    BellmanFord_Queue_init();
    BellmanFord_Queue(source);

	graph_traits < Graph >::vertex_iterator vi, vend;
	i = 0;
	for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
		if (distance[*vi] != Distance[i++])
			__ERROR("distance error");
	}
    BellmanFord_Queue_end();
}
