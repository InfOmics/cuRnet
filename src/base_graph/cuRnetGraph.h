#ifndef __CURNET_H__
#define __CURNET_H__

#ifdef __cplusplus
extern "C"
{
#endif 


#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP
cuRnet_graph(
DataFrame dataFrame
);

// [[Rcpp::export]]
SEXP cuRnet_graph_output(
SEXP graph_ptr
);


// [[Rcpp::export]]
RcppExport SEXP i_cuRnet_graph(SEXP dataFrameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< DataFrame >::type dataFrame(dataFrameSEXP);
    rcpp_result_gen = Rcpp::wrap(cuRnet_graph(dataFrame));
    return rcpp_result_gen;
END_RCPP
}




#ifdef __cplusplus
}
#endif 

#endif
