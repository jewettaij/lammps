#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <array>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "math_eigen.h"



// ----------------- Unit Tests for MathEigen::LambdaLanczos ----------------
//
// This code was taken from
// https://github.com/mrcdr/lambda-lanczos/blob/master/test/lambda_lanczos_test.cpp


using namespace MathEigen;

template<typename T>
using vector = std::vector<T>;

template<typename T>
using complex = std::complex<T>;

/*void sig_digit_test() {
  std::cout << std::endl << "-- Significant decimal digit test --" << std::endl;
  std::cout << "float " << MathEigen::sig_decimal_digit<float>() << " "
	    << MathEigen::minimum_effective_decimal<float>() << std::endl;
  std::cout << "double " << MathEigen::sig_decimal_digit<double>() << " "
	    << MathEigen::minimum_effective_decimal<double>() << std::endl;
  std::cout << "long double " << MathEigen::sig_decimal_digit<long double>() << " "
	    << MathEigen::minimum_effective_decimal<long double>() << std::endl;
}*/

template <typename T>
void vector_initializer(vector<T>& v);

template<>
void vector_initializer(vector<double>& v) {
  std::mt19937 mt(1);
  std::uniform_real_distribution<double> rand(-1.0, 1.0);

  size_t n = v.size();
  for(size_t i = 0;i < n;i++) {
    v[i] = rand(mt);
  }
}

template<>
void vector_initializer(vector<complex<double>>& v) {
  std::mt19937 mt(1);
  std::uniform_real_distribution<double> rand(-1.0, 1.0);

  size_t n = v.size();
  for(size_t i = 0;i < n;i++) {
    v[i] = std::complex<double>(rand(mt), rand(mt));
  }
}

TEST(UNIT_TEST, INNER_PRODUCT) {
  complex<double> c1(1.0, 3.0);
  complex<double> c2(2.0, 4.0);

  vector<complex<double>> v1{3.0, c1};
  vector<complex<double>> v2{3.0, c2};

  auto result = MathEigen::inner_prod(v1, v2);
  complex<double> correct(23.0, -2.0);

  EXPECT_DOUBLE_EQ(correct.real(), result.real());
  EXPECT_DOUBLE_EQ(correct.imag(), result.imag());
}

TEST(UNIT_TEST, L1_NORM) {
  complex<double> c1(1.0, 3.0);
  complex<double> c2(-1.0, -1.0);

  vector<complex<double>> v{c1, c2};

  EXPECT_DOUBLE_EQ(sqrt(10.0)+sqrt(2.0), MathEigen::l1_norm(v));
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX) {
  const size_t n = 3;
  double matrix[n][n] = { {2.0, 1.0, 1.0},
                          {1.0, 2.0, 1.0},
			  {1.0, 1.0, 2.0} };
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    }
  };

  LambdaLanczos<double> engine(matmul, n, true);
  engine.init_vector = vector_initializer<double>;
  engine.eigenvalue_offset = 6.0;

  double eigvalue;
  vector<double> eigvec(1); // The size will be enlarged automatically
  engine.run(eigvalue, eigvec);


  auto sign = eigvec[0]/std::abs(eigvec[0]);
  vector<double> correct_eigvec(n);
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] = sign*1.0/sqrt(3.0);
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue*engine.eps*10));
  }
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_NOT_FIX_RANDOM_SEED) {
  const size_t n = 3;
  double matrix[n][n] = { {2.0, 1.0, 1.0},
                          {1.0, 2.0, 1.0},
			  {1.0, 1.0, 2.0} };
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    }
  };

  LambdaLanczos<double> engine(matmul, n, true);
  engine.eigenvalue_offset = 6.0;

  double eigvalue;
  vector<double> eigvec(1); // The size will be enlarged automatically
  engine.run(eigvalue, eigvec);


  auto sign = eigvec[0]/std::abs(eigvec[0]);
  vector<double> correct_eigvec(n);
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] = sign*1.0/sqrt(3.0);
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue*engine.eps*10));
  }
}

TEST(DIAGONALIZE_TEST, DYNAMIC_MATRIX) {
  const size_t n = 10;

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(size_t i = 0;i < n-1;i++) {
      out[i] += -1.0*in[i+1];
      out[i+1] += -1.0*in[i];
    }

    //    out[0] += -1.0*in[n-1]; // This corresponds to
    //    out[n-1] += -1.0*in[0]; // periodic boundary condition
  };
  /*
    This lambda is equivalent to applying following n by n matrix

      0  -1   0       ..      0
     -1   0  -1       ..      0
      0  -1   0       ..      0
      0     ..        ..      0
      0     ..        0  -1   0
      0     ..       -1   0  -1
      0     ..        0  -1   0

      Its smallest eigenvalue is -2*cos(pi/(n+1)).
   */

  LambdaLanczos<double> engine(matmul, n);
  engine.init_vector = vector_initializer<double>;
  engine.eps = 1e-14;
  engine.eigenvalue_offset = -10.0;
  double eigvalue;
  vector<double> eigvec(n);
  engine.run(eigvalue, eigvec);

  double correct_eigvalue = -2.0*cos(M_PI/(n+1));
  auto sign = eigvec[0]/std::abs(eigvec[0]);
  vector<double> correct_eigvec(n);
  for(size_t i = 0;i < n;i++) {
    correct_eigvec[i] = sign * std::sin((i+1)*M_PI/(n+1));
  }
  MathEigen::normalize(correct_eigvec);

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue*engine.eps*10));
  }
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_USE_COMPLEX_TYPE) {
  const size_t n = 3;
  complex<double> matrix[n][n] = { {2.0, 1.0, 1.0},
				   {1.0, 2.0, 1.0},
				   {1.0, 1.0, 2.0} };
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    }
  };

  LambdaLanczos<complex<double>> engine(matmul, n, true);
  engine.init_vector = vector_initializer<complex<double>>;
  double eigvalue;
  vector<complex<double>> eigvec(n);
  engine.run(eigvalue, eigvec);


  vector<complex<double>> correct_eigvec(n);
  complex<double> phase_factor = std::exp(complex<double>(0.0, 1.0)*std::arg(eigvec[0]));
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] = 1.0 / std::sqrt(n) * phase_factor;
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), std::abs(correct_eigvalue*engine.eps*10));
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), std::abs(correct_eigvalue*engine.eps*10));
  }
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_USE_COMPLEX_TYPE_NOT_FIX_RANDOM_SEED) {
  const size_t n = 3;
  complex<double> matrix[n][n] = { {2.0, 1.0, 1.0},
				   {1.0, 2.0, 1.0},
				   {1.0, 1.0, 2.0} };
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    }
  };

  LambdaLanczos<complex<double>> engine(matmul, n, true);
  double eigvalue;
  vector<complex<double>> eigvec(n);
  engine.run(eigvalue, eigvec);


  vector<complex<double>> correct_eigvec(n);
  complex<double> phase_factor = std::exp(complex<double>(0.0, 1.0)*std::arg(eigvec[0]));
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] = 1.0 / std::sqrt(n) * phase_factor;
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), std::abs(correct_eigvalue*engine.eps*10));
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), std::abs(correct_eigvalue*engine.eps*10));
  }
}

TEST(DIAGONALIZE_TEST, HERMITIAN_MATRIX) {
  const size_t n = 3;
  const auto I_ = complex<double>(0.0, 1.0);
  complex<double> matrix[n][n] = { { 0.0, I_  , 1.0},
				   { -I_, 0.0 , I_ },
				   { 1.0, -I_ , 0.0} };
  /* Its eigenvalues are {-2, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    }
  };

  LambdaLanczos<complex<double>> engine(matmul, n);
  engine.init_vector = vector_initializer<complex<double>>;
  double eigvalue;
  vector<complex<double>> eigvec(n);
  engine.run(eigvalue, eigvec);


  vector<complex<double>> correct_eigvec { 1.0, I_, -1.0 };
  MathEigen::normalize(correct_eigvec);
  complex<double> phase_factor = std::exp(complex<double>(0.0, 1.0)*std::arg(eigvec[0]));
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] *= phase_factor;
  }

  double correct_eigvalue = -2.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), std::abs(correct_eigvalue*engine.eps*10));
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), std::abs(correct_eigvalue*engine.eps*10));
  }
}

template <typename T, typename RE>
void generate_random_matrix(T** a, vector<T>& eigvec, T& eigvalue,
			    size_t n, size_t rand_n, RE eng) {
  const T min_eigvalue = 1.0;
  std::uniform_int_distribution<size_t> dist_index(0, n-1);
  std::uniform_real_distribution<double> dist_angle(0.0, 2*M_PI);
  std::uniform_real_distribution<double> dist_element(min_eigvalue, n*10);

  T max_eigvalue = min_eigvalue;
  size_t max_eig_index = 0;
  for(size_t i = 0;i < n;i++) {
    a[i] = a[0]+n*i;
    std::fill(a[i], a[i]+n, 0.0);
    a[i][i] = dist_element(eng);
    if(a[i][i] > max_eigvalue) {
      max_eigvalue = a[i][i];
      max_eig_index = i;
    }
  }

  eigvalue = max_eigvalue;

  /* Eigenvector corresponding to the maximum eigenvalue */
  std::fill(eigvec.begin(), eigvec.end(), T());
  eigvec[max_eig_index] = 1.0;

  for(size_t i = 0;i < rand_n;i++) {
    size_t k = dist_index(eng);
    size_t l = dist_index(eng);
    while(k == l) {
      l = dist_index(eng);
    }

    T theta = dist_angle(eng);

    T c = std::cos(theta);
    T s = std::sin(theta);
    T akk = a[k][k];
    T akl = a[k][l];
    T all = a[l][l];

    for(size_t i = 0;i < n;i++) {
      T aki_next = c*a[k][i] - s*a[l][i];
      a[l][i]    = s*a[k][i] + c*a[l][i];
      a[k][i] = aki_next;
    }

    /* Symmetrize */
    for(size_t i = 0;i < n;i++) {
      a[i][k] = a[k][i];
      a[i][l] = a[l][i];
    }

    a[k][k] = c*(c*akk - s*akl) - s*(c*akl - s*all);
    a[k][l] = s*(c*akk - s*akl) + c*(c*akl - s*all);
    a[l][k] = a[k][l];
    a[l][l] = s*(s*akk + c*akl) + c*(s*akl + c*all);


    T vk_next = c*eigvec[k] - s*eigvec[l];
    eigvec[l] = s*eigvec[k] + c*eigvec[l];
    eigvec[k] = vk_next;
  }
}

TEST(DIAGONALIZE_TEST, RANDOM_SYMMETRIC_MATRIX) {
  const size_t n = 10;

  double** matrix = new double*[n];
  matrix[0] = new double[n*n];
  for(size_t i = 0;i < n;i++) {
    matrix[i] = matrix[0]+n*i;
  }

  vector<double> correct_eigvec(n);
  double correct_eigvalue = 0.0;

  generate_random_matrix(matrix, correct_eigvec, correct_eigvalue,
			 n, n*10, std::mt19937(1));

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    }
  };

  LambdaLanczos<double> engine(matmul, n, true);
  engine.init_vector = vector_initializer<double>;
  engine.eps = 1e-14;
  double eigvalue;
  vector<double> eigvec(n);
  engine.run(eigvalue, eigvec);

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  auto sign = eigvec[0]/std::abs(eigvec[0]);
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i]*sign, eigvec[i], std::abs(correct_eigvalue*engine.eps*10));
  }

  delete[] matrix[0];
  delete[] matrix;
}






// --------------------- Unit Tests for MathEigen::Jacobi -------------------



// This code was taken from:
// https://github.com/jewettaij/jacobi_pd/blob/master/tests/test_jacobi.cpp
// That code used "assert()" statements instead of google-test.
// In order to make this work with google-test (gtest), I just changed the
// "assert()" statements to google-style "ASSERT_TRUE()" statements.
// (But I am probably not using as many google test features as I should.)
//
// By default, only matrices of type double** are used for testing.
// The many #ifdefs in this ugly code were originally used to compile different
// versions of this test code to test different kinds of matrix implementations
// (for example, double **, vector<vector<double>>, double[3][3], etc...),
// If I have time, I'll clean this up and get rid of the #ifdef statements.
// -Andrew 2020-9-06

using std::cout;
using std::cerr;
using std::endl;
using std::setprecision;
using std::vector;
using std::array;


// This code works with various types of C++ matrices (for example,
// double **, vector<vector<double>> array<array<double,5>,5>).
// I use "#if defined" statements to test different matrix types.
// For some of these (eg. array<array<double,5>,5>), the size of the matrix
// must be known at compile time.  I specify that size now.
#if defined USE_ARRAY_OF_ARRAYS
const int NF=5;  //(the array size must be known at compile time)
#elif defined USE_C_FIXED_SIZE_ARRAYS
const int NF=5;  //(the array size must be known at compile time)
#endif


// @brief  Are two numbers "similar"?
template<typename Scalar>
inline static bool Similar(Scalar a, Scalar b,
                           Scalar eps=1.0e-06,
                           Scalar ratio=1.0e-06,
                           Scalar ratio_denom=1.0)
{
  return ((std::abs(a-b)<=std::abs(eps))
          ||
          (std::abs(ratio_denom)*std::abs(a-b)
           <=
           std::abs(ratio)*0.5*(std::abs(a)+std::abs(b))));
}

/// @brief  Are two vectors (containing n numbers) similar?
template<typename Scalar, typename Vector>
inline static bool SimilarVec(Vector a, Vector b, int n,
                              Scalar eps=1.0e-06,
                              Scalar ratio=1.0e-06,
                              Scalar ratio_denom=1.0)
{
  for (int i = 0; i < n; i++)
    if (not Similar(a[i], b[i], eps, ratio, ratio_denom))
      return false;
  return true;
}

/// @brief  Are two vectors (or their reflections) similar?
template<typename Scalar, typename Vector>
inline static bool SimilarVecUnsigned(Vector a, Vector b, int n,
                                      Scalar eps=1.0e-06,
                                      Scalar ratio=1.0e-06,
                                      Scalar ratio_denom=1.0)
{
  if (SimilarVec(a, b, n, eps))
    return true;
  else {
    for (int i = 0; i < n; i++)
      if (not Similar(a[i], -b[i], eps, ratio, ratio_denom))
        return false;
    return true;
  }
}


/// @brief  Multiply two matrices A and B, store the result in C. (C = AB).

template<typename Matrix, typename ConstMatrix>
void mmult(ConstMatrix A, //<! input array
           ConstMatrix B, //<! input array
           Matrix C,      //<! store result here
           int m,      //<! number of rows of A
           int n=0,    //<! optional: number of columns of B (=m by default)
           int K=0     //<! optional: number of columns of A = num rows of B (=m by default)
           )
{
  if (n == 0) n = m; // if not specified, then assume the matrices are square
  if (K == 0) K = m; // if not specified, then assume the matrices are square

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      C[i][j] = 0.0;

  // perform matrix multiplication
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < K; k++)
        C[i][j] += A[i][k] * B[k][j];
}



/// @brief
///Sort the rows of a matrix "evec" by the numbers contained in "eval"
///(This is a simple O(n^2) sorting method, but O(n^2) is a lower bound anyway.)
///This is the same as the Jacobi::SortRows(), but that function is private.
template<typename Scalar, typename Vector, typename Matrix>
void
SortRows(Vector eval,
         Matrix evec,
         int n,
         bool sort_decreasing=true,
         bool sort_abs=false)
{
  for (int i = 0; i < n-1; i++) {
    int i_max = i;
    for (int j = i+1; j < n; j++) {
      if (sort_decreasing) {
        if (sort_abs) { //sort by absolute value?
          if (std::abs(eval[j]) > std::abs(eval[i_max]))
            i_max = j;
        }
        else if (eval[j] > eval[i_max])
          i_max = j;
      }
      else {
        if (sort_abs) { //sort by absolute value?
          if (std::abs(eval[j]) < std::abs(eval[i_max]))
            i_max = j;
        }
        else if (eval[j] < eval[i_max])
          i_max = j;
      }
    }
    std::swap(eval[i], eval[i_max]); // sort "eval"
    for (int k = 0; k < n; k++)
      std::swap(evec[i][k], evec[i_max][k]); // sort "evec"
  }
}



/// @brief  Generate a random orthonormal n x n matrix

template<typename Scalar, typename Matrix>
void GenRandOrth(Matrix R,
                 int n,
                 std::default_random_engine &rand_generator)
{
  std::normal_distribution<Scalar> gaussian_distribution(0,1);
  std::vector<Scalar> v(n);

  for (int i = 0; i < n; i++) {
    // Generate a vector, "v", in a random direction subject to the constraint
    // that it is orthogonal to the first i-1 rows-vectors of the R matrix.
    Scalar rsq = 0.0;
    while (rsq == 0.0) {
      // Generate a vector in a random direction
      // (This works because we are using a normal (Gaussian) distribution)
      for (int j = 0; j < n; j++)
        v[j] = gaussian_distribution(rand_generator);

      //Now subtract from v, the projection of v onto the first i-1 rows of R.
      //This will produce a vector which is orthogonal to these i-1 row-vectors.
      //(They are already normalized and orthogonal to each other.)
      for (int k = 0; k < i; k++) {
        Scalar v_dot_Rk = 0.0;
          for (int j = 0; j < n; j++)
            v_dot_Rk += v[j] * R[k][j];
        for (int j = 0; j < n; j++)
          v[j] -= v_dot_Rk * R[k][j];
      }
      // check if it is linearly independent of the other vectors and non-zero
      rsq = 0.0;
      for (int j = 0; j < n; j++)
        rsq += v[j]*v[j];
    }
    // Now normalize the vector
    Scalar r_inv = 1.0 / std::sqrt(rsq);
    for (int j = 0; j < n; j++)
      v[j] *= r_inv;
    // Now copy this vector to the i'th row of R
    for (int j = 0; j < n; j++)
      R[i][j] = v[j];
  } //for (int i = 0; i < n; i++)
} //void GenRandOrth()



/// @brief  Generate a random symmetric n x n matrix, M.
/// This function generates random numbers for the eigenvalues ("evals_known")
/// as well as the eigenvectors ("evecs_known"), and uses them to generate M.
/// The "eval_magnitude_range" argument specifies the the base-10 logarithm
/// of the range of eigenvalues desired.  The "n_degeneracy" argument specifies
/// the number of repeated eigenvalues desired (if any).
/// @returns  This function does not return a value.  However after it is
///           invoked, the M matrix will be filled with random numbers.
///           Additionally, the "evals" and "evecs" arguments will contain
///           the eigenvalues and eigenvectors (one eigenvector per row)
///           of the matrix.  Later, they can be compared with the eigenvalues
///           and eigenvectors calculated by Jacobi::Diagonalize()

template <typename Scalar, typename Vector, typename Matrix>
void GenRandSymm(Matrix M,       //<! store the matrix here
                 int n,          //<! matrix size
                 Vector evals,   //<! store the eigenvalues of here
                 Matrix evecs,   //<! store the eigenvectors here
                 std::default_random_engine &rand_generator,//<! makes random numbers
                 Scalar min_eval_size=0.1, //<! minimum possible eigenvalue size
                 Scalar max_eval_size=10.0,//<! maximum possible eigenvalue size
                 int n_degeneracy=1//<!number of repeated eigevalues(1disables)
                 )
{
  ASSERT_TRUE(n_degeneracy <= n);
  std::uniform_real_distribution<Scalar> random_real01;
  std::normal_distribution<Scalar> gaussian_distribution(0, max_eval_size);
  bool use_log_uniform_distribution = false;
  if (min_eval_size > 0.0)
    use_log_uniform_distribution = true;
  #if defined USE_VECTOR_OF_VECTORS
  vector<vector<Scalar> > D(n, vector<Scalar>(n));
  vector<vector<Scalar> > tmp(n, vector<Scalar>(n));
  #elif defined USE_ARRAY_OF_ARRAYS
  array<array<Scalar, NF>, NF> D;
  array<array<Scalar, NF>, NF> tmp;
  #elif defined USE_C_FIXED_SIZE_ARRAYS
  Scalar D[NF][NF], tmp[NF][NF];
  #else
  #define USE_C_POINTER_TO_POINTERS
  Scalar  **D, **tmp;
  Alloc2D(n, n, &D);
  Alloc2D(n, n, &tmp);
  #endif

  // Randomly generate the eigenvalues
  for (int i = 0; i < n; i++) {
    if (use_log_uniform_distribution) {
      // Use a "log-uniform distribution" (a.k.a. "reciprocal distribution")
      // (This is a way to specify numbers with a precise range of magnitudes.)
      ASSERT_TRUE((min_eval_size > 0.0) && (max_eval_size > 0.0));
      Scalar log_min = std::log(std::abs(min_eval_size));
      Scalar log_max = std::log(std::abs(max_eval_size));
      Scalar log_eval = (log_min + random_real01(rand_generator)*(log_max-log_min));
      evals[i] = std::exp(log_eval);
      // also consider both positive and negative eigenvalues:
      if (random_real01(rand_generator) < 0.5)
        evals[i] = -evals[i];
    }
    else {
      evals[i] = gaussian_distribution(rand_generator);
    }
  }

  // Does the user want us to force some of the eigenvalues to be the same?
  if (n_degeneracy > 1) {
    int *permutation = new int[n]; //a random permutation from 0...n-1
    for (int i = 0; i < n; i++)
      permutation[i] = i;
    std::shuffle(permutation, permutation+n, rand_generator);
    for (int i = 1; i < n_degeneracy; i++) //set the first n_degeneracy to same
      evals[permutation[i]] = evals[permutation[0]];
    delete [] permutation;
  }

  // D is a diagonal matrix whose diagonal elements are the eigenvalues
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      D[i][j] = ((i == j) ? evals[i] : 0.0);

  // Now randomly generate the (transpose of) the "evecs" matrix
  GenRandOrth<Scalar, Matrix>(evecs, n, rand_generator); //(will transpose it later)

  // Construct the test matrix, M, where M = Rt * D * R

  // Original code:
  //mmult(evecs, D, tmp, n);  // <--> tmp = Rt * D
  // Unfortunately, C++ guesses the types incorrectly.  Must manually specify:
  // #ifdefs making the code ugly again:
  #if defined USE_VECTOR_OF_VECTORS
  mmult<vector<vector<Scalar> >&, const vector<vector<Scalar> >&>
  #elif defined USE_ARRAY_OF_ARRAYS
  mmult<array<array<Scalar,NF>,NF>&, const array<array<Scalar,NF>,NF>&>
  #elif defined USE_C_FIXED_SIZE_ARRAYS
  mmult<Scalar (*)[NF], Scalar (*)[NF]>
  #else
  mmult<Scalar**, Scalar const *const *>
  #endif
       (evecs, D, tmp, n);

  for (int i = 0; i < n-1; i++)
    for (int j = i+1; j < n; j++)
      std::swap(evecs[i][j], evecs[j][i]); //transpose "evecs"

  // Original code:
  //mmult(tmp, evecs, M, n);
  // Unfortunately, C++ guesses the types incorrectly.  Must manually specify:
  // #ifdefs making the code ugly again:
  #if defined USE_VECTOR_OF_VECTORS
  mmult<vector<vector<Scalar> >&, const vector<vector<Scalar> >&>
  #elif defined USE_ARRAY_OF_ARRAYS
  mmult<array<array<Scalar,NF>,NF>&, const array<array<Scalar,NF>,NF>&>
  #elif defined USE_C_FIXED_SIZE_ARRAYS
  mmult<Scalar (*)[NF], Scalar (*)[NF]>
  #else
  mmult<Scalar**, Scalar const *const *>
  #endif
       (tmp, evecs, M, n);
  //at this point M = Rt*D*R (where "R"="evecs")

  #if defined USE_C_POINTER_TO_POINTERS
  Dealloc2D(&D);
  Dealloc2D(&tmp);
  #endif
} // GenRandSymm()



template <typename Scalar>
void TestJacobi(int n, //<! matrix size
                int n_matrices=100, //<! number of matrices to test
                Scalar min_eval_size=0.1,  //<! minimum possible eigenvalue sizw
                Scalar max_eval_size=10.0, //<! maximum possible eigenvalue size
                int n_tests_per_matrix=1, //<! repeat test for benchmarking?

                int n_degeneracy=1, //<! repeated eigenvalues?
                unsigned seed=0, //<! random seed (if 0 then use the clock)
                Scalar eps=1.0e-06
                )
{
  bool test_code_coverage = false;
  if (n_tests_per_matrix < 1) {
    cout << "-- Testing code-coverage --" << endl;
    test_code_coverage = true;
    n_tests_per_matrix = 1;
  }
  cout << endl << "-- Diagonalization test (real symmetric)  --" << endl;

  // construct a random generator engine using a time-based seed:

  if (seed == 0) // if the caller did not specify a seed, use the system clock
    seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine rand_generator(seed);



  // Create an instance of the Jacobi diagonalizer, and allocate the matrix
  // we will test it on, as well as the arrays that will store the resulting
  // eigenvalues and eigenvectors.
  // The way we do this depends on what version of the code we are using.
  // This is controlled by "#if defined" statements.
  
  #if defined USE_VECTOR_OF_VECTORS

  Jacobi<Scalar,
         vector<Scalar>&,
         vector<vector<Scalar> >&,
         const vector<vector<Scalar> >& > ecalc(n);

  // allocate the matrix, eigenvalues, eigenvectors
  vector<vector<Scalar> > M(n, vector<Scalar>(n));
  vector<vector<Scalar> > evecs(n, vector<Scalar>(n));
  vector<vector<Scalar> > evecs_known(n, vector<Scalar>(n));
  vector<Scalar> evals(n);
  vector<Scalar> evals_known(n);
  vector<Scalar> test_evec(n);

  #elif defined USE_ARRAY_OF_ARRAYS

  n = NF;
  cout << "Testing std::array (fixed size).\n"
    "(Ignoring first argument, and setting matrix size to " << n << ")" << endl;

  Jacobi<Scalar,
         array<Scalar, NF>&,
         array<array<Scalar, NF>, NF>&,
         const array<array<Scalar, NF>, NF>&> ecalc(n);

  // allocate the matrix, eigenvalues, eigenvectors
  array<array<Scalar, NF>, NF> M;
  array<array<Scalar, NF>, NF> evecs;
  array<array<Scalar, NF>, NF> evecs_known;
  array<Scalar, NF> evals;
  array<Scalar, NF> evals_known;
  array<Scalar, NF> test_evec;

  #elif defined USE_C_FIXED_SIZE_ARRAYS

  n = NF;
  cout << "Testing C fixed size arrays.\n"
    "(Ignoring first argument, and setting matrix size to " << n << ")" << endl;
  Jacobi<Scalar,
         Scalar*,
         Scalar (*)[NF],
         Scalar const (*)[NF]> ecalc(n);

  // allocate the matrix, eigenvalues, eigenvectors
  Scalar M[NF][NF];
  Scalar evecs[NF][NF];
  Scalar evecs_known[NF][NF];
  Scalar evals[NF];
  Scalar evals_known[NF];
  Scalar test_evec[NF];

  #else
 
  #define USE_C_POINTER_TO_POINTERS

  // Note: Normally, you would just use this to instantiate Jacobi:
  // Jacobi<Scalar, Scalar*, Scalar**, Scalar const*const*> ecalc(n);
  // -------------------------
  // ..but since Jacobi manages its own memory using new and delete, I also want
  // to test that the copy constructors, copy operators, and destructors work.
  // The following lines do this:
  Jacobi<Scalar, Scalar*, Scalar**, Scalar const*const*> ecalc_test_mem1;
  ecalc_test_mem1.SetSize(n);
  Jacobi<Scalar, Scalar*, Scalar**, Scalar const*const*> ecalc_test_mem2(2);
  // test the = operator
  ecalc_test_mem2 = ecalc_test_mem1;
  // test the copy constructor
  Jacobi<Scalar, Scalar*, Scalar**, Scalar const*const*> ecalc(ecalc_test_mem2);
  // allocate the matrix, eigenvalues, eigenvectors
  Scalar **M, **evecs, **evecs_known;
  Alloc2D(n, n, &M);
  Alloc2D(n, n, &evecs);
  Alloc2D(n, n, &evecs_known);
  Scalar *evals = new Scalar[n];
  Scalar *evals_known = new Scalar[n];
  Scalar *test_evec = new Scalar[n];

  #endif


  // --------------------------------------------------------------------
  // Now, generate random matrices and test Jacobi::Diagonalize() on them.
  // --------------------------------------------------------------------

  for(int imat = 0; imat < n_matrices; imat++) {

    // Create a randomly generated symmetric matrix.
    //This function generates random numbers for the eigenvalues ("evals_known")
    //as well as the eigenvectors ("evecs_known"), and uses them to generate M.

    #if defined USE_VECTOR_OF_VECTORS
    GenRandSymm<Scalar, vector<Scalar>&, vector<vector<Scalar> >&>
    #elif defined USE_ARRAY_OF_ARRAYS
    GenRandSymm<Scalar, array<Scalar,NF>&, array<array<Scalar,NF>,NF>&>
    #elif defined USE_C_FIXED_SIZE_ARRAYS
    GenRandSymm<Scalar, Scalar*, Scalar (*)[NF]>
    #else
    GenRandSymm<Scalar, Scalar*, Scalar**>
    #endif
               (M,
                n,
                evals_known,
                evecs_known,
                rand_generator,
                min_eval_size,
                max_eval_size,
                n_degeneracy);

    // Sort the matrix evals and eigenvector rows:
    // Original code:
    //SortRows<Scalar>(evals_known, evecs_known, n);
    // Unfortunately, C++ guesses the types incorrectly. Must use #ifdefs again:
    #if defined USE_VECTOR_OF_VECTORS
    SortRows<Scalar, vector<Scalar>&, vector<vector<Scalar> >&>
    #elif defined USE_ARRAY_OF_ARRAYS
    SortRows<Scalar, array<Scalar,NF>&, array<array<Scalar,NF>,NF>&>
    #elif defined USE_C_FIXED_SIZE_ARRAYS
    SortRows<Scalar, Scalar*, Scalar (*)[NF]>
    #else
    SortRows<Scalar, Scalar*, Scalar**>
    #endif
            (evals_known, evecs_known, n);


    if (n_matrices == 1) {
      cout << "Eigenvalues (after sorting):\n";
      for (int i = 0; i < n; i++)
        cout << evals_known[i] << " ";
      cout << "\n";
      cout << "Eigenvectors (rows) which are known in advance:\n";
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
          cout << evecs_known[i][j] << " ";
        cout << "\n";
      }
      cout << "  (The eigenvectors calculated by Jacobi::Diagonalize() should match these.)\n";
    }

    for (int i_test = 0; i_test < n_tests_per_matrix; i_test++) {

      if (test_code_coverage) {

        // test SORT_INCREASING_ABS_EVALS:
        #if defined USE_VECTOR_OF_VECTORS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 vector<Scalar>&,
                                 vector<vector<Scalar> >&,
                                 const vector<vector<Scalar> >& >::SORT_INCREASING_ABS_EVALS);
        #elif defined USE_ARRAY_OF_ARRAYS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 array<Scalar,NF>&,
                                 array<array<Scalar,NF>,NF>&,
                                 const array<array<Scalar,NF>,NF>&>::SORT_INCREASING_ABS_EVALS);
        #elif defined USE_C_FIXED_SIZE_ARRAYS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 Scalar*,
                                 Scalar (*)[NF],
                                 Scalar const (*)[NF]>::SORT_INCREASING_ABS_EVALS);
        #else
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 Scalar*,
                                 Scalar**,
                                 Scalar const*const*>::SORT_INCREASING_ABS_EVALS);
        #endif

        for (int i = 1; i < n; i++)
          ASSERT_TRUE(std::abs(evals[i-1])<=std::abs(evals[i]));

        // test SORT_DECREASING_ABS_EVALS:
        #if defined USE_VECTOR_OF_VECTORS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 vector<Scalar>&,
                                 vector<vector<Scalar> >&,
                                 const vector<vector<Scalar> >& >::SORT_DECREASING_ABS_EVALS);
        #elif defined USE_ARRAY_OF_ARRAYS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 array<Scalar,NF>&,
                                 array<array<Scalar,NF>,NF>&,
                                 const array<array<Scalar,NF>,NF>&>::SORT_DECREASING_ABS_EVALS);
        #elif defined USE_C_FIXED_SIZE_ARRAYS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 Scalar*,
                                 Scalar (*)[NF],
                                 Scalar const (*)[NF]>::SORT_DECREASING_ABS_EVALS);
        #else
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 Scalar*,
                                 Scalar**,
                                 Scalar const*const*>::SORT_DECREASING_ABS_EVALS);
        #endif

        for (int i = 1; i < n; i++)
          ASSERT_TRUE(std::abs(evals[i-1])>=std::abs(evals[i]));

        // test SORT_INCREASING_EVALS:
        #if defined USE_VECTOR_OF_VECTORS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 vector<Scalar>&,
                                 vector<vector<Scalar> >&,
                                 const vector<vector<Scalar> >& >::SORT_INCREASING_EVALS);
        #elif defined USE_ARRAY_OF_ARRAYS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 array<Scalar,NF>&,
                                 array<array<Scalar,NF>,NF>&,
                                 const array<array<Scalar,NF>,NF>&>::SORT_INCREASING_EVALS);
        #elif defined USE_C_FIXED_SIZE_ARRAYS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 Scalar*,
                                 Scalar (*)[NF],
                                 Scalar const (*)[NF]>::SORT_INCREASING_EVALS);
        #else
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 Scalar*,
                                 Scalar**,
                                 Scalar const*const*>::SORT_INCREASING_EVALS);
        #endif
        for (int i = 1; i < n; i++)
          ASSERT_TRUE(evals[i-1] <= evals[i]);

        // test DO_NOT_SORT
        #if defined USE_VECTOR_OF_VECTORS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 vector<Scalar>&,
                                 vector<vector<Scalar> >&,
                                 const vector<vector<Scalar> >& >::DO_NOT_SORT);
        #elif defined USE_ARRAY_OF_ARRAYS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 array<Scalar,NF>&,
                                 array<array<Scalar,NF>,NF>&,
                                 const array<array<Scalar,NF>,NF>&>::DO_NOT_SORT);
        #elif defined USE_C_FIXED_SIZE_ARRAYS
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 Scalar*,
                                 Scalar (*)[NF],
                                 Scalar const (*)[NF]>::DO_NOT_SORT);
        #else
        ecalc.Diagonalize(M,
                          evals,
                          evecs,
                          Jacobi<Scalar,
                                 Scalar*,
                                 Scalar**,
                                 Scalar const*const*>::DO_NOT_SORT);
        #endif

      } //if (test_code_coverage)


      // Now (finally) calculate the eigenvalues and eigenvectors
      int n_sweeps = ecalc.Diagonalize(M, evals, evecs);

      if ((n_matrices == 1) && (i_test == 0)) {
        cout <<"Jacobi::Diagonalize() ran for "<<n_sweeps<<" iters (sweeps).\n";
        cout << "Eigenvalues calculated by Jacobi::Diagonalize()\n";
        for (int i = 0; i < n; i++)
          cout << evals[i] << " ";
        cout << "\n";
        cout << "Eigenvectors (rows) calculated by Jacobi::Diagonalize()\n";
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++)
            cout << evecs[i][j] << " ";
          cout << "\n";
        }
      }

      ASSERT_TRUE(SimilarVec(evals, evals_known, n, eps*max_eval_size, eps));
      //Check that each eigenvector satisfies Mv = λv
      // <-->  Σ_b  M[a][b]*evecs[i][b] = evals[i]*evecs[i][b]   (for all a)
      for (int i = 0; i < n; i++) {
        for (int a = 0; a < n; a++) {
          test_evec[a] = 0.0;
          for (int b = 0; b < n; b++)
            test_evec[a] += M[a][b] * evecs[i][b];
          ASSERT_TRUE(Similar(test_evec[a],
                              evals[i] * evecs[i][a],
                              eps,  // tolerance (absolute difference)
                              eps*max_eval_size, // tolerance ratio (numerator)
                              evals_known[i] // tolerance ration (denominator)
                              ));
        }
      }

    } //for (int i_test = 0; i_test < n_tests_per_matrix; i++)

  } //for(int imat = 0; imat < n_matrices; imat++) {

  #if defined USE_C_POINTER_TO_POINTERS
  Dealloc2D(&M);
  Dealloc2D(&evecs);
  Dealloc2D(&evecs_known);
  delete [] evals;
  delete [] evals_known;
  delete [] test_evec;
  #endif

} //TestJacobi()




TEST_JACOBI()
{
    TestJacobi(2, 100, 0.1, 10);
    TestJacobi(5, 100, 0.01, 100);
    TestJacobi(20, 40, 1e-05, 1e+05);
    TestJacobi(50, 10, 1e-09, 1e+09);
}
