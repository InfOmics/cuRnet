#' Create a cuRnet graph object from a 3-column \code{DataFrame}.
#' The \code{DataFrame} represents edges in the form (source vertex, destination vertex, weight). 
#' First two colums are of type \code{charatcter}, weights are of type \code{numeric}.
#' Weights column is optional, but must be specified for algorithms that require the information, such as SSSP.
#' 
#' @import
#' @export
#' 
#' @details
#' 
#' @examples
#' 
#' @examples
#' \dontrun{
#' library(STRINGdb)
#' library(igraph)
#' library(cuRnet)
#' ss <- STRINGdb$new( version="10", species=9606, score_threshold=900)
#' g <- ss$get_graph()
#' from <- unique(ends(g,E(g))[,1])[1:10]
#' x <- data.frame("from" = ends(g,E(g))[,1], "to" = ends(g,E(g))[,2], "score" = E(g)$combined_score/1000)
#' cgraph <- cuRnet_graph(x)
#' }
#' 
cuRnet_graph <- function(dataFrame) {
  .Call('i_cuRnet_graph', PACKAGE = 'cuRnet', dataFrame)
}



#' Single Source Shortest Paths: distances and predecessors.
#' This function computes shortest paths from a series of vertices.
#' For each source vertex, shortest paths to every vertex in the graph are computed.
#' The function returns distances and predecessors.
#'
#' @param graph A cuRnet graph object created with \code{cuRnet_graph}.
#' @param from A \code{CharacterVector} with the names of the source vertices.
#' @return A list of two \code{NumericMatrix} indexed by "distances" and "predecessors". Rows are source vertices and colums are network vertices. A entry is: "distances" the distance along the shortest path from the source to the destination vertex; "predecessors" a minimal predecessor of the vertex and along the shortest path.
#' 
#' @import
#' @export
#' 
#' @details
#' 
#' 
#' @examples
#' \dontrun{
#' library(STRINGdb)
#' library(igraph)
#' library(cuRnet)
#' ss <- STRINGdb$new( version="10", species=9606, score_threshold=900)
#' g <- ss$get_graph()
#' from <- V(g)$name[1:10]
#' x <- data.frame("from" = ends(g,E(g))[,1], "to" = ends(g,E(g))[,2], "score" = E(g)$combined_score/1000)
#' cg <- cuRnet_graph(x)
#' ret <- cuRnet_sssp(g, from)
#' ret[["distances"]]
#' ret[["predecessors"]]
#' }
#' 
cuRnet_sssp <- function(graph, from) {
    .Call('i_cuRnet_sssp', PACKAGE = 'cuRnet', graph, from)
}


#' Single Source Shortest Paths: distances only.
#' This function computes shortest paths from a series of vertices.
#' For each source vertex, shortest paths to every vertex in the graph are computed.
#' The function returns only distances.
#'
#' @param graph A cuRnet graph object created with \code{cuRnet_graph}.
#' @param from A \code{CharacterVector} with the names of the source vertices.
#' @return A \code{NumericMatrix} where rows are source vertices and colums are network vertices. An entry is the distance along the shortest path from the source to the destination vertex.
#' 
#' @import
#' @export
#' 
#' @details
#' 
#' 
#' @examples
#' \dontrun{
#' library(STRINGdb)
#' library(igraph)
#' library(cuRnet)
#' ss <- STRINGdb$new( version="10", species=9606, score_threshold=900)
#' g <- ss$get_graph()
#' from <- V(g)$name[1:10]
#' x <- data.frame("from" = ends(g,E(g))[,1], "to" = ends(g,E(g))[,2], "score" = E(g)$combined_score/1000)
#' cg <- cuRnet_graph(x)
#' ret <- cuRnet_sssp_dists(cg, from)
#' ret[["distances"]][1,] 
#' ret[["predecessors"]][1,]
#' ret[1,]
#' }
#' 
cuRnet_sssp_dists <- function(graph, from) {
    .Call('i_cuRnet_sssp_dists', PACKAGE = 'cuRnet', graph, from)
}

#' \code{cuRnet_scc}: Strongly Connected Components
#' This function computes strongly connected components membership for every vertex of the input graph.
#'
#' @param graph A cuRnet graph object created with \code{cuRnet_graph}.
#' @return A \code{NumericMatrix} of 1 row and number of columns equal to the number of graph vertices. Each cell reports the identifier of the connected component associated with the corresponding vertex.
#' 
#' @import
#' @export
#' 
#' @details
#' 
#' 
#' @examples
#' \dontrun{
#' library(igraph)
#' library(cuRnet)
#' rg <- sample_fitness_pl(10000, 30000, 2.2, 2.3)
#' cdf <- data.frame( ends(rg, E(rg))[,1], ends(rg, E(rg))[,2] )
#' colnames(cdf) <- c("from", "to")
#' cg <- cuRnet_graph(cdf)
#' cc <- cuRnet_scc(cg)
#' length(unique(cc[1,])) #number of found strongly connected components
#' }
#' 
cuRnet_scc <- function(graph) {
  .Call('i_cuRnet_scc', PACKAGE = 'cuRnet', graph)
}

#' Breadth-first search. 
#' This function traverses the graph via a breadth-first search from a given set of source vertices, and returns depth of visited nodes.
#'
#' @param graph A cuRnet graph object created with \code{cuRnet_graph}.
#' @param sources The lists of source vertices from which to start BFSs. Per every source, one BFS is performed.
#' @return A \code{NumericMatrix} having a number of rows equal to the number of source vertices, and a number of columns equal to the total number of vertices of th einput graph. Every row correspondo to a specific source vertex, and row cell reports the depth from the given source to the correspondig graph vertex.
#' 
#' @import
#' @export
#' 
#' @details
#' 
#' 
#' @examples
#' \dontrun{
#' library(igraph)
#' library(cuRnet)
#' rg <- sample_fitness_pl(100, 1000, 2.2, 2.3)
#' cdf <- data.frame( ends(rg, E(rg))[,1], ends(rg, E(rg))[,2] )
#' colnames(cdf) <- c("from", "to")
#' sources <- union(cdf$from, cdf$to)[1:20]
#' cg <- cuRnet_graph(cdf)
#' bfs <- cuRnet_bfs(cg, sources)
#' bfs[1,]
#' }
#' 
cuRnet_bfs <- function(graph, sources) {
  .Call('i_cuRnet_bfs', PACKAGE = 'cuRnet', graph, sources)
}
