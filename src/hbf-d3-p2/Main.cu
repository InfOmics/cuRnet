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
#include "XLib.hpp"
#include "Host/GraphSSSP.hpp"
#include "Device/HBFGraph.cuh"
#include <vector>
#include "../PCSF_types.h"


bool check_paths(int from,
                 int to,
                 DistPath *dists_paths){
int current, pred;

for (int second = from+1; second<to; ++second) {
	std::vector<int> seen;
	current=second;
	seen.push_back(current);
	if(dists_paths[current].parent == -1){
		dists_paths[current].parent = current;
	}
	pred = dists_paths[current].parent;
	while(pred!=current){
		if( std::find(seen.begin(),seen.end(),pred) != seen.end()){
			//for(vector<int>::iterator it = seen.begin(); it!=seen.end(); it++){
			//	std::cout<<(*it)<<" ";
			//}
			std::cout<<pred;
			std::cout<<std::endl;
			return false;
		}

		seen.push_back(pred);
		current=pred;
		pred=dists_paths[current].parent;
		if(pred == -1){
			pred = current;
		}
	}
}
return true;
};



int main(int argc, char** argv) {
    using namespace boost;

    GraphBasePCSF bgraph;
    std::cout  <<argv[1] << std::endl;
    //return 1;

    std::ifstream fin(argv[1]);

    std::string str;
    fin >> str >> str >> str;

    int count = -1;
    int max = -1;
    while (!fin.eof()) {
        int source, dest;
        float weight;
        fin >> source >> dest >> weight;

        //std::cout << source <<  " " << dest << " " <<  weight << std::endl;
        count++;
        if (source > max)
            max = source;
        if (dest > max)
            max = dest;
        //if (weight < 0)
        //    std::cout << weight << std::endl;
	   boost::add_edge(source, dest, std::abs(weight), bgraph);
    }
    max++;
    fin.close();


	int V =  num_vertices(bgraph);
	int E = num_edges(bgraph);

    int num_undirected = E * 2;

	GraphSSSP graph(V, num_undirected, EdgeType::DIRECTED);
    graph.COOSize = num_undirected;

    auto weight_array = new float[num_undirected];

	using    EdgeIterator = graph_traits<GraphBasePCSF>::edge_iterator;
	graph_traits<GraphBasePCSF>::edge_descriptor edge;

	using PropMap = boost::property_map<GraphBasePCSF, boost::edge_weight_t>::type;
	PropMap edge_weight_map = get(boost::edge_weight, bgraph);

	int source;
	int dest;
	float weight;
	EdgeIterator it, end_it;

    int i = 0;
	for (boost::tie(it, end_it) = boost::edges(bgraph); it != end_it; ++it) {
		edge = *it;
		source = boost::source(edge, bgraph) - 1;
		dest = boost::target(edge, bgraph) - 1;
		weight = edge_weight_map[edge];

		graph.COO_Edges[i][0] = source;
		graph.COO_Edges[i][1] = dest;
		weight_array[i] = std::abs(weight);
		graph.COO_Edges[E + i][0] = dest;
		graph.COO_Edges[E + i][1] = source;
		weight_array[E + i] = std::abs(weight);
		i++;
	}

    graph.ToCSR(weight_array);

    HBFGraph devGraph(graph, false);
    devGraph.copyToDevice();


    std::cout << "init" << std::endl;
	//for(int source = 0; source < 50; source++){
		//weight_t *distss = new weight_t[graph.V];
	    //int *paths = new int[graph.V];
        auto d_data = new DistPath[graph.V];


	    devGraph.WorkEfficient_PCSF(0, d_data, false);
	    //--------------------------------------------------

	    weight_t *h_distss;
	    int *h_paths;

	    graph.BellmanFord_Queue_init();

	    //for (int i = 1; i < 500; i++) {
		graph.BellmanFord_Queue(0);
		graph.BellmanFord_Result(h_distss, h_paths);
	    //    graph.BellmanFord_Queue_reset();
	    //}

	    for (int i = 0; i < V; i++) {
    		if (d_data[i].dist != h_distss[i])
    		    std::cout << "!dist " << i << "\t\t "<< d_data[i].dist << " " <<  h_distss[i] << std::endl;
		//if (paths[i] != h_paths[i])
	        std::cout << "path " << i << "\t\t" << d_data[i].parent << " " <<  h_paths[i] << std::endl;
	    }
	    //    std::cout << i << "\t" << distss[i] << "\t" << paths[i] << std::endl;
        std::cout << "correct <>" << std::endl;

		if(! check_paths(source, 50, d_data)){
			std::cout<<"NOT P"<<std::endl;
		}

	    //delete[] distss;
	    //delete[] paths;
        delete[] d_data;
	//}

 delete[] weight_array;
}
