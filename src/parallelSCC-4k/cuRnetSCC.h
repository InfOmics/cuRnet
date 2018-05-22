#ifndef __CURNET_H__
#define __CURNET_H__

#ifdef __cplusplus
extern "C"
{
#endif 


#include <Rcpp.h>

using namespace Rcpp;


// [[Rcpp::export]]
NumericMatrix
cuRnet_scc(
SEXP graphPtr
);

NumericMatrix
cuRnet_bfs(
SEXP graphPtr, Rcpp::CharacterVector source
);

// [[Rcpp::export]]
RcppExport SEXP i_cuRnet_scc(SEXP graphSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type graphPtr(graphSEXP);
    rcpp_result_gen = Rcpp::wrap(cuRnet_scc(graphPtr));
    return rcpp_result_gen;
END_RCPP
}


// [[Rcpp::export]]
RcppExport SEXP i_cuRnet_bfs(SEXP graphSEXP, SEXP source) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type graphPtr(graphSEXP);
    Rcpp::traits::input_parameter< CharacterVector >::type src(source);
    rcpp_result_gen = Rcpp::wrap(cuRnet_bfs(graphPtr, src));
    return rcpp_result_gen;
END_RCPP
}




#ifdef __cplusplus
}
#endif 

#endif
