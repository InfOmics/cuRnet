#include "cuModule.cuh"

#include "Device/HBFGraph.cuh"
#include "timer.h"

#include <vector>

bool check_paths_dp(
std::vector<int>& terminals,
DistPath *dp
){
int current, pred;
for (std::vector<int>::iterator second = terminals.begin(); second!=terminals.end(); ++second) {
	std::vector<int> seen;
	current=*second; 
	seen.push_back(current);
	if(dp[current].parent == -1){
		dp[current].parent = current;
	}
	pred = dp[current].parent;
	while(pred!=current){
		if( std::find(seen.begin(),seen.end(),pred) != seen.end()){
			for(std::vector<int>::iterator it = seen.begin(); it!=seen.end(); it++){
				std::cout<<(*it)<<" ";
			}
			std::cout<<pred;
			std::cout<<std::endl;
			return false;
		}
		seen.push_back(pred);
		current=pred; 
		pred=dp[current].parent; 
		if(pred == -1){
			pred = current;
		}
	} 
}
return true;
};







void cu_sssp(
	curnet::base_graph *graph, 
	std::vector<int> &sources, 
	std::vector< std::vector<float>> &costs,
	std::vector< std::vector<int>> &parents){

	HBFGraph devGraph(*graph, false);
	devGraph.copyToDevice();

	int V = graph->v();
	//std::cout<<V<<std::endl;
	

	DistPath *dp = new DistPath[V];


	int source;

//TIMEHANDLE start;
//double ptime = 0.0;

	for(int i=0; i<sources.size(); i++){

		source = sources[i];

		//start = start_time();
		devGraph.WorkEfficient_PCSF(source, dp);
		//ptime += end_time(start);

		//check_paths_dp(sources,dp);

		for(int j=0; j<V; j++){
			costs[i][j] = dp[j].dist;
			parents[i][j] = dp[j].parent;
		}
	}

//std::cout<<"CUDA runtime "<<ptime<<std::endl;
	delete [] dp;
};



void cu_sssp_dists(
	curnet::base_graph *graph, 
	std::vector<int> &sources, 
	std::vector< std::vector<float>> &costs){

	HBFGraph devGraph(*graph, false);
	devGraph.copyToDevice();

	int V = graph->v();
	//std::cout<<V<<std::endl;
	

	DistPath *dp = new DistPath[V];


	int source;

//TIMEHANDLE start;
//double ptime = 0.0;

	for(int i=0; i<sources.size(); i++){

		source = sources[i];

		//start = start_time();
		devGraph.WorkEfficient_PCSF(source, dp, false);
		//ptime += end_time(start);

		//check_paths_dp(sources,dp);

		for(int j=0; j<V; j++){
			costs[i][j] = dp[j].dist;
		}
	}

//std::cout<<"CUDA runtime "<<ptime<<std::endl;
	delete [] dp;
};
