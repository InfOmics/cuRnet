#include "Rcpp.h"
using namespace Rcpp;
#include <map>
#include <unordered_map>


#include "base_graph.hpp"

//#include "cuModuleGraph.cuh"

#include "cuRnetGraph.h"

#include "../timer.h"

//XPtr< TestClass >

using namespace std;
using namespace curnet;

// [[Rcpp::export]]
SEXP
cuRnet_graph(
DataFrame dataFrame
)
{
	Rcpp::CharacterVector sources = dataFrame[0];
	Rcpp::CharacterVector destinations = dataFrame[1];
	Rcpp::NumericVector weights;
	if(dataFrame.size() > 2)
		weights = dataFrame[2];

	unordered_map< string, int> node_to_id;
	//unordered_map< int, string> id_to_node;
	vector< string> id_to_node;
	unordered_map< string, int>::iterator node_it;
	string s;
	int nof_E = sources.size();
	
	unique_ptr<base_graph> graph(new base_graph(nof_E, Directed));
	
	int cidx = 0;
	
	int2* coo = graph->edges();
	float* w = graph->weights();
	for(int i=0; i<sources.size(); i++){
		int2 edge;
		node_it = node_to_id.find( Rcpp::as<std::string>(sources[i]));
		if(node_it == node_to_id.end()){
			edge.x = (int)node_to_id.size();
			//id_to_node.insert(pair< int, string>((int)node_to_id.size(), Rcpp::as<string>(sources[i])));
			node_to_id.insert(pair< string, int>(Rcpp::as<string>(sources[i]), (int)node_to_id.size()));
		}
		else
		{
			edge.x = node_it->second;
		}
		node_it = node_to_id.find( Rcpp::as<std::string>(destinations[i]));

		if(node_it == node_to_id.end()){
			edge.y = (int)node_to_id.size();
			//id_to_node.insert(pair< int, string>((int)node_to_id.size(), Rcpp::as<string>(destinations[i])));
			node_to_id.insert(pair< string, int>(Rcpp::as<string>(destinations[i]), (int)node_to_id.size()));
		}
	   else
	   {
		   edge.y = node_it->second;
		}
		
		coo[i] = edge;
		if(dataFrame.size()>2)
			w[i] = std::abs(weights[i]);
		else
			w[i] = 1;
		
	}
	
	id_to_node.resize(node_to_id.size());
	for(auto v : node_to_id) id_to_node[v.second] = v.first;
		

	int nof_V = node_to_id.size();
	graph->v(nof_V);
	if(!graph->to_csr()) std::cout << "to_csr fail!" << std::endl;
	
	graph->node_to_id() = node_to_id;
	graph->id_to_node() = id_to_node;


	//scc4k::SCCGraph graph(nof_V, nof_E, GDirection::DIRECTED);
	//graph.COOSize = nof_E;

	//int source, dest;
	
	//std::memcpy(graph.COO_Edges, COO, sizeof(int)*2*nof_E);
	//delete[] COO;
	//graph.ToCSR();
	
	
	Rcpp::XPtr<base_graph> ptr(graph.release(), true);
	return ptr;
}
