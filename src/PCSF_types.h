#pragma once
/*
// Loading the required libraries
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <vector>
#include <boost/limits.hpp>


//using namespace boost;
using namespace std;


// NodePCSF class
class NodePCSF
{  public:
  vector <int> children;
  int father;
  int size;
  double price;
  NodePCSF(){};
  NodePCSF(const NodePCSF & other){
    father=other.father;
    children= other.children;}

  NodePCSF & operator= (const NodePCSF & other){
    father=other.father;
    children= other.children;
    return *this;
  }

};

// Properties of Vertices in the network
struct VertexPropertiesPCSF  {
  VertexPropertiesPCSF() :c(0) {}
  VertexPropertiesPCSF(string const & name) : name(name),c(0){}
  string name;
  double c;
};

// Properties of the graph that is used within the BOOST
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexPropertiesPCSF, boost::property < boost::edge_weight_t, double > > GraphBasePCSF;
typedef boost::graph_traits<GraphBasePCSF>::vertex_iterator vertex_iterator;
typedef boost::graph_traits<GraphBasePCSF>::out_edge_iterator edge_iterator;
typedef boost::graph_traits<GraphBasePCSF>::edge_iterator graph_edge_iterator;
typedef boost::graph_traits<GraphBasePCSF>::edge_descriptor Edge;
typedef boost::graph_traits<GraphBasePCSF>::vertex_descriptor Vertex;
*/
