AC_INIT([cuRnet], 0.5.2)
AC_CONFIG_AUX_DIR([tools])

AC_PROG_CC
AC_PROG_CXX

AC_ARG_WITH([cuda-home],
     AC_HELP_STRING([--with-cuda-home=cuda_home],
          [specify the root of your cuda install, e.g. /usr/local/cuda]),
     [CUDA_HOME="$withval"])

AC_ARG_WITH([r-include],
     AC_HELP_STRING([--with-r-include=r_include],
          [specify the full path to your R header files, e.g. /usr/share/R/include]),
     [R_INC="$withval"])

AC_ARG_WITH([r-lib],
     AC_HELP_STRING([--with-r-lib=r-lib],
          [specify the full path to your R shared libraries, e.g. /usr/lib64/R/lib]),
     [R_LIB="$withval"])

AC_ARG_WITH([single-precision],
     AC_HELP_STRING([--with-single-precision=single-precision],
          [specify whether to compile in single-precision mode, e.g. 1]),
     [SNGL="$withval"])

AC_MSG_CHECKING("environment variable CUDA_HOME")
if test -z "${CUDA_HOME}"; then
     AC_MSG_RESULT("CUDA_HOME not set; using default /usr/local/cuda")
     CUDA_HOME="/usr/local/cuda"
else
     AC_MSG_RESULT("using CUDA_HOME=${CUDA_HOME}")
fi

AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
if test -z "${HAS_NVCC}"; then
     AC_MSG_ERROR("cannot find nvcc; check CUDA install")
else
     AC_MSG_RESULT("nvcc found")
fi

AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [BIT64="yes"])
if test -z "${BIT64}"; then
     AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
     CUDA_LIB_NAME="/lib"
else
     AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
     CUDA_LIB_NAME="/lib64"
fi

CUDA_INC="-I${CUDA_HOME}/include"

CUDA_LIB_PATH=${CUDA_HOME}${CUDA_LIB_NAME}
CUDA_LIB="\"-rpath ${CUDA_LIB_PATH} -L${CUDA_LIB_PATH}\""

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
     AC_MSG_ERROR("could not determine R_HOME")
else
     CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`
     if test -z "${R_INC}"; then
          R_INC=`"${R_HOME}/bin/R" CMD config --cppflags`
          R_INC="\"${R_INC}\""
     else
          R_INC="-I${R_INC}"
     fi
     if test -z "${R_LIB}"; then
          R_LIB="-L${R_HOME}/lib -lR"
     else
          R_LIB="-L${R_LIB} -lR"
     fi
fi


echo "================================="
RCPP_INC=`Rscript -e "Rcpp:::CxxFlags()"`
RCPP_LD=`Rscript -e 'Rcpp:::LdFlags()'`
R_INC="${R_INC} ${RCPP_INC}"
R_LIB="${R_LIB} ${RCPP_LD}"
echo ${R_INC}
echo ${R_LIB}


`${CUDA_HOME}/bin/nvcc getDeviceInfo.cu -Wno-deprecated-gpu-targets -o getDeviceInfo`
DAC_ARCH=`./getDeviceInfo | cut -d";" -f1`
DAC_SM=`./getDeviceInfo | cut -d";" -f2`
echo ${DAC_ARCH}
echo ${DAC_SM}

`${CUDA_HOME}/bin/nvcc getResidentThreads.cu -Wno-deprecated-gpu-targets -o getResidentThreads`
DAC_RES_THREADS=`./getResidentThreads 0`
echo ${DAC_RES_THREADS}
echo "================================="


AC_SUBST(CUDA_HOME)
AC_SUBST(CUDA_INC)
AC_SUBST(CUDA_LIB)

AC_SUBST(R_HOME)
AC_SUBST(R_INC)
AC_SUBST(R_LIB)

AC_SUBST(DAC_ARCH)
AC_SUBST(DAC_SM)
AC_SUBST(DAC_RES_THREADS)

AC_SUBST(CPICFLAGS)
AC_SUBST(SNGL)


AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT
