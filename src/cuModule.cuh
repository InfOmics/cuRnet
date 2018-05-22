#pragma once

#include "Host/GraphSSSP.hpp"

#include <base_graph.hpp>

void cu_sssp(
	curnet::base_graph *graph, 
	std::vector<int> &sources, 
	std::vector< std::vector<float>> &costs,
	std::vector< std::vector<int>> &parents);



void cu_sssp_dists(
	curnet::base_graph *graph, 
	std::vector<int> &sources, 
	std::vector< std::vector<float>> &costs);
