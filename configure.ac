dnl package name and version
AC_INIT([fRLR], 1.0)

dnl check gsl lib
#: ${GSL_LIBS=`gsl-config --libs`}
#if test -z "${GSL_LIBS}"; then
#  echo "could find not lgsl and lgslcblas"
#  exit 1
#fi

dnl check gsl include
#: ${GSL_CFLAGS=`gsl-config --cflags`}
#if test -z "${GSL_CFLAGS}"; then
#  echo "could not find the cflags of gsl"
#  exit 1
#fi

AC_PATH_PROG([GSL_CONFIG], [gsl-config])
if test "${GSL_CONFIG}" != ""; then
   GSL_CFLAGS=`${GSL_CONFIG} --cflags`
   GSL_LIBS=`${GSL_CONFIG} --libs`
else
   AC_MSG_ERROR([gsl-config not found, please make sure you have installed GSL.])
fi

dnl check openmp.h
AC_PROG_CC([cc cl gcc])
AC_CHECK_HEADER(omp.h)
if test "${ac_cv_header_omp_h}" = yes; then
  OPENMP_CFLAGS="-fopenmp"
else
  OPENMP_CFLAGS=""
  echo "Not find openmp.h"
fi


AC_SUBST(GSL_LIBS)
AC_SUBST(GSL_CFLAGS)
AC_SUBST(OPENMP_CFLAGS)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
