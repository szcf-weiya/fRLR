dnl package name and version
AC_INIT([fRLR], 1.0)

dnl check gsl lib
: ${GSL_LIBS=`gsl-config --libs`}
if test -z "${GSL_LIBS}"; then
  echo "could find not lgsl and lgslcblas"
  exit 1
fi

dnl check gsl include
: ${GSL_CFLAGS=`gsl-config --cflags`}
if test -z "${GSL_CFLAGS}"; then
  echo "could not find the cflags of gsl"
  exit 1
fi

dnl check openmp.h
AC_CHECK_HEADER(omp.h)
if test "${ac_cv_header_omp_h}" = yes; then
  omp_CFLAGS="-fopenmp"
  fRLR_CFLAGS="${GSL_CFLAGS} ${omp_CFLAGS}"
else
  fRLR_CFLAGS="${GSL_CFLAGS}"
  echo "Not find openmp.h"
fi


AC_SUBST(GSL_LIBS)
AC_SUBST(fRLR_CFLAGS)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
