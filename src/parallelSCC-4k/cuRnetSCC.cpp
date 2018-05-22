#include "Rcpp.h"
using namespace Rcpp;
#include <map>
#include <unordered_map>

#include "cuModuleSCC.cuh"

#include "cuRnetSCC.h"

#include "../timer.h"

// [[Rcpp::export]]
NumericMatrix
cuRnet_scc(
SEXP graph_ptr
){
	Rcpp::XPtr<curnet::base_graph> graph(graph_ptr);
	int nof_V = graph->v();
	
	std::vector<int> colors(nof_V);
	colors.resize(nof_V);

	cu_scc(*graph, colors);

	Rcpp::NumericMatrix ret(1, colors.size());
	int idx = 0;
	for( const auto& sm_pair : graph->node_to_id() )
	{
		ret(0, idx) = colors[sm_pair.second];
		idx++;
	}

	CharacterVector cnames(nof_V);
	for(int node = 0; node < nof_V; ++node)
	{
		cnames[node] = (graph->id_to_node()[node]);
	}
	colnames(ret) = cnames;

	return ret;
}

// [[Rcpp::export]]
NumericMatrix
cuRnet_bfs(
SEXP graph_ptr, Rcpp::CharacterVector source
){
	Rcpp::XPtr<curnet::base_graph> graph(graph_ptr);
	curnet::base_graph* g = graph;
	
	int nof_V = g->v();
	std::vector<int> sources(source.size());
	for(int i=0; i<source.size(); i++){
		sources[i] = (g->node_to_id())[Rcpp::as<std::string>(source[i])];
	}
  

	std::vector<std::vector<float>> distance(source.size());
	for(int i = 0; i < distance.size(); i++) distance[i].resize(nof_V);
//	TIMEHANDLE start = start_time();
	cu_bfs(g, distance, sources);
//	double ptime = end_time(start);
//	std::cout << "Total exec time: " << ptime << std::endl;
	
	CharacterVector cnames(nof_V);
	Rcpp::NumericMatrix ret(distance.size(), nof_V);
	for(int node = 0; node < nof_V; ++node)
	{
		cnames[node] = (g->id_to_node()[node]);
		for(int i = 0; i < distance.size(); i++)
			ret(i, node) = distance[i][node];
			
	}
	rownames(ret) = source;
	colnames(ret) = cnames;

	return ret;
}
