#include<Rcpp.h>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<sstream>
#include<vector>
#include<string>
#include<stdlib.h>
#include<stdio.h>

#include<gsl/gsl_vector.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_combination.h>
#include<gsl/gsl_statistics.h>
#include<gsl/gsl_fit.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_multifit.h>
#include<gsl/gsl_cdf.h>
#include<gsl/gsl_permutation.h>
#include<gsl/gsl_blas.h>
#include<omp.h>

using namespace std;

//[[Rcpp::export]]
Rcpp::List frlr1(SEXP X, SEXP Y, SEXP COV)
{
  return 0;
}

//[[Rcpp::export]]
Rcpp::List frlr2(SEXP R_X, SEXP R_idx1, SEXP R_idx2, SEXP R_Y, SEXP R_COV)
{
  // gsl_matrix is row-major order

  // convert data type
  Rcpp::NumericVector X(R_X);
  Rcpp::NumericVector Y(R_Y);
  Rcpp::NumericVector COV(R_COV);
  Rcpp::IntegerVector idx1(R_idx1);
  Rcpp::IntegerVector idx2(R_idx2);

  int nrow = Y.size();
  int nX = X.size();
  int ncol = nX/nrow;

  double *Xarray = (double*)malloc(sizeof(double)*nX);
  gsl_matrix_view Xview = gsl_matrix_view_array(Xarray, nrow, ncol);

  double *Yarray = (double*)malloc(sizeof(double)*nrow);
  gsl_vector_view Yview = gsl_vector_view_array(Yarray, nrow);

  int n = idx1.size();
  int nCOV = COV.size();
  int COV_COL = nCOV/nrow;

  double *COVarray = (double*)malloc(sizeof(double)*nCOV);
  gsl_matrix_view COVview = gsl_matrix_view_array(COVarray, nrow, COV_COL);

  gsl_matrix *b = gsl_matrix_alloc(nrow, COV_COL+2);
  gsl_matrix *B = gsl_matrix_alloc(COV_COL+2, COV_COL+2);
  gsl_matrix *invB = gsl_matrix_alloc(COV_COL+2, COV_COL+2);

  gsl_vector *covariate_vec = gsl_vector_alloc(nrow);
  for (int ip = 0; ip < COV_COL; ip++)
  {
    gsl_matrix_get_col(covariate_vec, &COVview.matrix, ip);
    gsl_matrix_set_col(b, 1+ip, covariate_vec);
  }
  gsl_vector_free(covariate_vec);
  free(COVarray);

  // intercept
  gsl_vector *x0 = gsl_vector_alloc(nrow);
  gsl_vector_set_all(x0, 1);
  gsl_matrix_set_col(b, 0, x0);

  // degree of freedom
  int df = nrow - COV_COL - 2 - 1;

  // cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
  // cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";

  vector<int> r1, r2;
  vector<double> r1_p, r2_p;

  gsl_permutation *permutation_B = gsl_permutation_alloc(B->size1);
  int status;

  // B = b'b
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, b, b, 0.0, B);

  // inv(B) by LU decomposition
  gsl_linalg_LU_decomp(B, permutation_B, &status);
  gsl_linalg_LU_invert(B, permutation_B, invB);

  # pragma omp parallel for schedule(dynamic) // faster!!!
  for (int j = 0; j < n; j++)
  {
    // the identical terms
    gsl_vector *x1 = gsl_vector_alloc(nrow);
    gsl_vector *x2 = gsl_vector_alloc(nrow);
    gsl_matrix_get_col(x1, &Xview.matrix, idx1[j]);
    gsl_matrix_get_col(x2, &Xview.matrix, idx2[j]);

    double A_1i_11, A_1i_12, A_1i_22;
    double a11, a12, a21, a22, a_det;
    gsl_matrix *invA_1i = gsl_matrix_alloc(2, 2);
    gsl_matrix *a_1i = gsl_matrix_alloc(nrow, 2);
    gsl_matrix *V_1i = gsl_matrix_alloc(COV_COL+2, 2);
    gsl_matrix *invB_mul_V_1i = gsl_matrix_alloc(COV_COL+2, 2);
    gsl_matrix *m_tmp = gsl_matrix_alloc(2, 2);
    gsl_matrix *B_1 = gsl_matrix_alloc(2, 2);
    gsl_matrix *invD = gsl_matrix_alloc(2, 2);
    gsl_matrix *m_tmp2 = gsl_matrix_alloc(COV_COL+2, 2);
    gsl_matrix *m_tmp3 = gsl_matrix_alloc(2, 2);
    gsl_matrix *m_tmp4 = gsl_matrix_alloc(COV_COL+2, 2);
    gsl_matrix *invXX_11 = gsl_matrix_alloc(2, 2);
    gsl_matrix *invXX_22 = gsl_matrix_alloc(COV_COL+2, COV_COL+2);
    gsl_matrix *invXX_21 = gsl_matrix_alloc(COV_COL+2, 2);
    gsl_vector *XY_1 = gsl_vector_alloc(2);
    gsl_vector *XY_2 = gsl_vector_alloc(COV_COL+2);
    gsl_vector *beta_1 = gsl_vector_alloc(2);
    gsl_vector *beta_2 = gsl_vector_alloc(COV_COL+2);
    gsl_vector *Yhat = gsl_vector_alloc(nrow);
    double rss, zscore1, zscore2, pvalue1, pvalue2;

    // A_1i
    gsl_blas_ddot(x1, x1, &A_1i_11);
    gsl_blas_ddot(x1, x2, &A_1i_12);
    gsl_blas_ddot(x2, x2, &A_1i_22);

    // invA_1i
    a_det = A_1i_11*A_1i_22-A_1i_12*A_1i_12;
    a11 = A_1i_22/a_det;
    a12 = -A_1i_12/a_det;
    a22 = A_1i_11/a_det;
    gsl_matrix_set(invA_1i, 0, 0, a11);
    gsl_matrix_set(invA_1i, 1, 1, a22);
    gsl_matrix_set(invA_1i, 0, 1, a12);
    gsl_matrix_set(invA_1i, 1, 0, a12);

    // construct a_1i
    gsl_matrix_set_col(a_1i, 0, x1);
    gsl_matrix_set_col(a_1i, 1, x2);

    // V_1i = b'a_1i
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, b, a_1i, 0.0, V_1i);

    // invB_mul_V_1i
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invB, V_1i, 0.0, invB_mul_V_1i);

    // B_1 = V_1i' mul invB mul V_1i
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, invB_mul_V_1i, V_1i, 0.0, B_1);

    // D = I - B_1 * invA_1i
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, B_1, invA_1i, 0.0, m_tmp);

    a11 = gsl_matrix_get(m_tmp, 0, 0);
    a22 = gsl_matrix_get(m_tmp, 1, 1);
    a12 = gsl_matrix_get(m_tmp, 0, 1);

    // D is noy symmetric !!
    a21 = gsl_matrix_get(m_tmp, 1, 0);
    a11 += 1;
    a22 += 1;
    a_det = a11*a22-a12*a21;

    gsl_matrix_set(invD, 0, 0, a22/a_det);
    gsl_matrix_set(invD, 1, 1, a11/a_det);
    gsl_matrix_set(invD, 1, 0, -a21/a_det);
    gsl_matrix_set(invD, 0, 1, -a12/a_det);

    // invXX_11
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invA_1i, B_1, 0.0, m_tmp3);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, m_tmp3, invD, 0.0, m_tmp); // Do not replace m_tmp with m_tmp3 !!
    gsl_matrix_memcpy(invXX_11, invA_1i);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, m_tmp, invA_1i, 1.0, invXX_11);

    // invXX_22
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invB_mul_V_1i, invA_1i, 0.0, m_tmp2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, m_tmp2, invD, 0.0, m_tmp4);
    gsl_matrix_memcpy(invXX_22, invB);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, -1.0, m_tmp4, invB_mul_V_1i, 1.0, invXX_22);

    // invXX_21
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invB, V_1i, 0.0, m_tmp2);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, m_tmp2, invD, 0.0, m_tmp4);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, m_tmp4, invA_1i, 0.0, invXX_21);

    // X'Y
    gsl_blas_dgemv(CblasTrans, 1.0, a_1i, &Yview.vector, 0.0, XY_1);
    gsl_blas_dgemv(CblasTrans, 1.0, b, &Yview.vector, 0.0, XY_2);

    // beta
    gsl_blas_dgemv(CblasNoTrans, 1.0, invXX_11, XY_1, 0.0, beta_1);
    gsl_blas_dgemv(CblasTrans, 1.0, invXX_21, XY_2, 1.0, beta_1);
    gsl_blas_dgemv(CblasNoTrans, 1.0, invXX_21, XY_1, 0.0, beta_2);
    gsl_blas_dgemv(CblasNoTrans, 1.0, invXX_22, XY_2, 1.0, beta_2);

    // RSS
    gsl_blas_dgemv(CblasNoTrans, 1.0, a_1i, beta_1, 0.0, Yhat);
    gsl_blas_dgemv(CblasNoTrans, 1.0, b, beta_2, 1.0, Yhat);

    gsl_vector_sub(Yhat, &Yview.vector);

    gsl_blas_ddot(Yhat, Yhat, &rss);

    // zscore
    zscore1 = gsl_vector_get(beta_1, 0)/(sqrt(rss/df*gsl_matrix_get(invXX_11, 0, 0)));
    pvalue1 = 2*(zscore1 < 0 ? (1 - gsl_cdf_tdist_P(-zscore1, df)) : (1 - gsl_cdf_tdist_P(zscore1, df)));

    zscore2 = gsl_vector_get(beta_1, 1)/(sqrt(rss/df*gsl_matrix_get(invXX_11, 1, 1)));
    pvalue2 = 2*(zscore2 < 0 ? (1 - gsl_cdf_tdist_P(-zscore2, df)) : (1 - gsl_cdf_tdist_P(zscore2, df)));

    gsl_vector_free(x1);
    gsl_vector_free(x2);
    gsl_vector_free(XY_1);
    gsl_vector_free(XY_2);
    gsl_vector_free(beta_1);
    gsl_vector_free(beta_2);
    gsl_vector_free(Yhat);
    gsl_matrix_free(invXX_11);
    gsl_matrix_free(invXX_22);
    gsl_matrix_free(invXX_21);
    gsl_matrix_free(invA_1i);
    gsl_matrix_free(a_1i);
    gsl_matrix_free(V_1i);
    gsl_matrix_free(invB_mul_V_1i);
    gsl_matrix_free(m_tmp);
    gsl_matrix_free(B_1);
    gsl_matrix_free(invD);
    gsl_matrix_free(m_tmp2);
    gsl_matrix_free(m_tmp3);
    gsl_matrix_free(m_tmp4);
    #pragma omp critical
    {
      r1.push_back(idx1[j]);
      r2.push_back(idx2[j]);
      r1_p.push_back(pvalue1);
      r2_p.push_back(pvalue2);
    }
  }
  Rcpp::DataFrame output = Rcpp::DataFrame::create(Rcpp::Named("r1") = r1,
                                                    Rcpp::Named("r2") = r2,
                                                    Rcpp::Named("r1.p.value") = r1_p,
                                                    Rcpp::Named("r2.p.value") = r2_p);

  gsl_vector_free(x0);
  free(Xarray);
  free(Yarray);
  return output;
}
