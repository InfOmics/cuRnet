#include "Rcpp.h"
using namespace Rcpp;
#include <map>
#include <string>

using namespace std;

#include "Host/GraphSSSP.hpp"
#include "cuModule.cuh"

#include "cuRnet.h"

//' cuRnet Single Source Shortest Paths
//' @param dataFrame a dataframe with columns: from, to, weight
//' @param from list of sources 
//' @export
//' @return costs and predecessors from sources to every other vertex
// [[Rcpp::export]]
Rcpp::List
cuRnet_sssp(SEXP graph_ptr, Rcpp::CharacterVector from){
	
	Rcpp::XPtr<curnet::base_graph> graph(graph_ptr);
	curnet::base_graph* g = graph;
	
	std::unordered_map< string, int> node_to_id = g->node_to_id();
	std::unordered_map< string, int>::iterator node_it;
	std::vector<string> id_to_node = g->id_to_node();
	
	int nof_V = g->v();

	std::vector<int> from_ids(from.size());
	std::vector< std::vector<float>> costs(from.size());
	std::vector< std::vector<int>> parents(from.size());

	for(int i=0; i<from.size(); i++){
		from_ids[i] = (g->node_to_id())[Rcpp::as<std::string>(from[i])];
		costs[i].resize(nof_V);
		parents[i].resize(nof_V);
	}

	cu_sssp(g, from_ids, costs, parents);
	
	
	
	CharacterVector cnames = CharacterVector::create();
	Rcpp::NumericMatrix ret_costs(from.size(), nof_V);
	Rcpp::CharacterMatrix ret_parents(from.size(), nof_V);
	for(int node = 0; node < nof_V; ++node)
	{
		cnames.push_back(g->id_to_node()[node]);
		
		for(int i = 0; i < from.size(); i++)
		{
			ret_costs(i,node) = costs[i][node];
			if(parents[i][node] == -1) //NOT A VALID ID!!
				ret_parents(i,node) = "";
			else
				ret_parents(i,node) = g->id_to_node()[ parents[i][node] ];
		}
			
	}
	rownames(ret_costs) = from;
	colnames(ret_costs) = cnames;
	rownames(ret_parents) = from;
	colnames(ret_parents) = cnames;
	
	
	
/*
	Rcpp::NumericMatrix ret_costs(from.size(), nof_V);

	//Rcpp::NumericMatrix ret_parents(from.size(), nof_V);
	Rcpp::CharacterMatrix ret_parents(from.size(), nof_V);
	
	std::map<int, std::string> inverse_node_to_id;
	//std::map< std::string, int>::iterator node_it;
	for(node_it = node_to_id.begin(); node_it != node_to_id.end(); node_it++){	
		inverse_node_to_id.insert( std::pair<int, std::string>(node_it->second, node_it->first) );
	}

	for(int j=0; j<nof_V; j++){
		for(int i=0; i<from.size(); i++){
			ret_costs(i,j) = costs[i][j];
			//ret_parents(i,j) = parents[i][j];
			ret_parents(i,j) = inverse_node_to_id[ parents[i][j] ];
		}
	}

	rownames(ret_costs) = from;
	rownames(ret_parents) = from;
	CharacterVector cnames = CharacterVector::create();
	for(node_it = node_to_id.begin(); node_it!=node_to_id.end(); node_it++){
		cnames.push_back(node_it->first);
	}
	colnames(ret_costs) = cnames;
	colnames(ret_parents) = cnames;
	*/

	//Rcpp::List rets = Rcpp::List::create(ret_costs, ret_parents);
	Rcpp::List rets = Rcpp::List::create( Named("distances") = ret_costs, _["predecessors"] = ret_parents);
	return rets;
};





//' cuRnet Single Source Shortest Paths
//' @param dataFrame a dataframe with columns: from, to, weight
//' @param from list of sources 
//' @export
//' @return costs and predecessors from sources to every other vertex
// [[Rcpp::export]]
Rcpp::NumericMatrix
cuRnet_sssp_dists(SEXP graph_ptr, Rcpp::CharacterVector from){
	
	Rcpp::XPtr<curnet::base_graph> graph(graph_ptr);
	curnet::base_graph* g = graph;
	
	std::unordered_map< string, int> node_to_id = g->node_to_id();
	std::unordered_map< string, int>::iterator node_it;
	std::vector<string> id_to_node = g->id_to_node();
	
	int nof_V = g->v();

	std::vector<int> from_ids(from.size());
	std::vector< std::vector<float>> costs(from.size());

	for(int i=0; i<from.size(); i++){
		from_ids[i] = node_to_id[Rcpp::as<std::string>(from[i])];
		costs[i].resize(nof_V);
	}

	cu_sssp_dists(g, from_ids, costs);

//	std::cout<<from.size()<<"\t"<<nof_V<<"\t"<<node_to_id.size()<<std::endl;

	CharacterVector cnames = CharacterVector::create();
	Rcpp::NumericMatrix ret_costs(from.size(), nof_V);
	for(int node = 0; node < nof_V; ++node)
	{
		cnames.push_back(g->id_to_node()[node]);
		for(int i = 0; i < from.size(); i++)
			ret_costs(i,node) = costs[i][node];
			
	}
	rownames(ret_costs) = from;
	colnames(ret_costs) = cnames;

	return ret_costs;
};
