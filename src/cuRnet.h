#ifndef __CURNET_H__
#define __CURNET_H__

#ifdef __cplusplus
extern "C"
{
#endif 


#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
List
cuRnet_sssp(
SEXP graphPtr, 
CharacterVector from
);

// [[Rcpp::export]]
RcppExport SEXP i_cuRnet_sssp(SEXP graphSEXP, SEXP fromSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type graphPtr(graphSEXP);
    Rcpp::traits::input_parameter< CharacterVector >::type from(fromSEXP);
    rcpp_result_gen = Rcpp::wrap(cuRnet_sssp(graphPtr,from));
    return rcpp_result_gen;
END_RCPP
}



// [[Rcpp::export]]
NumericMatrix
cuRnet_sssp_dists(
SEXP graphPtr, 
CharacterVector from
);

// [[Rcpp::export]]
RcppExport SEXP i_cuRnet_sssp_dists(SEXP graphSEXP, SEXP fromSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type graphPtr(graphSEXP);
    Rcpp::traits::input_parameter< CharacterVector >::type from(fromSEXP);
    rcpp_result_gen = Rcpp::wrap(cuRnet_sssp_dists(graphPtr,from));
    return rcpp_result_gen;
END_RCPP
}


#ifdef __cplusplus
}
#endif 

#endif
