dnl package name and version
AC_INIT([fRLR], 1.0)

dnl check gsl
PKG_CHECK_MODULES([GSL], [gsl >= 1.6], [HAVEGSL="yes"], [HAVEGSL="no"])
if test "$HAVEGSL" = "yes"; then
   echo "GSL has been found."
   LIBS=$GSL_LIBS
else
   AC_MSG_ERROR([Cannot find GNU Scientific Library >= 1.6, please install it firstly.])
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
