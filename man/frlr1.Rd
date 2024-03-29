% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{frlr1}
\alias{frlr1}
\title{Fit Repeated Linear Regressions with One Variable}
\usage{
frlr1(R_X, R_Y, R_COV, num_threads = -1L)
}
\arguments{
\item{R_X}{the observation matrix}

\item{R_Y}{the response}

\item{R_COV}{common variables}

\item{num_threads}{number of threads for openmp. If it is -1 (default), it will use all possible threads.}
}
\value{
the fitting results for each regression.
}
\description{
Fit a set of linear regressions which differ only in one variable.
}
\examples{
set.seed(123)
X = matrix(rnorm(50), 10, 5)
Y = rnorm(10)
COV = matrix(rnorm(40), 10, 4)
frlr1(X, Y, COV)
}
