# fRLR

## Introduction

This R package aims to fit *Repeated Linear Regressions* in which there are some same terms.

## An Example

Suppose we want to fit a set of regressions which only differ in one variable. Specifically, denote the response variable as ![](https://latex.codecogs.com/gif.latex?y), and these regressions are as follows.

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Barray%7D%7Bll%7D%20y%26%5Csim%20x_1%20&plus;%20cov_1%20&plus;%20cov_2&plus;%5Cldots&plus;cov_m%5C%5C%20y%26%5Csim%20x_2%20&plus;%20cov_1%20&plus;cov_2&plus;%5Cldots&plus;cov_m%5C%5C%20%5Ccdot%20%26%5Csim%20%5Ccdots%5C%5C%20y%26%5Csim%20x_n%20&plus;%20cov_1%20&plus;cov_2&plus;%5Cldots&plus;cov_m%5C%5C%20%5Cend%7Barray%7D)

where ![](https://latex.codecogs.com/gif.latex?%24cov_i%2C%20i%3D1%2C%5Cldots%2C%20m%24) are the same variables among these regressions.

Intuitively, we can finish this task by using a simple loop in R code.

```
model = vector(mode='list', length=n)
for (i in 1:n)
{
  ...
  model[[i]] = lm(y~x)
}
```

However, it is not efficient in that situation. Due to the same variables, there are some same calculations among different regressions. In order to reduce the cost of computation, I write this `fRLR` package.

Let me take a simulation example to show how to use this R package.

```
## use fRLR package
library(fRLR)
set.seed(123)
X = matrix(rnorm(50), 10, 5)
Y = rnorm(10)
COV = matrix(rnorm(40), 10, 4)
frlr1(X, Y, COV)

## use simple loop
res = matrix(nrow = 0, ncol = 2)
for (i in 1:ncol(X))
{
  mat = cbind(X[,i], COV)
  df = as.data.frame(mat)
  model = lm(Y~., data = df)
  tmp = c(i, summary(model)$coefficients[2, 4])
  res = rbind(res, tmp)
}
```

Then we can obtain the following results

![](https://stats.hohoweiya.xyz/images//frlr_ex1.png)

To show the speed of my package, let me increase the number of regressions, and compare the time duration with simple-loop method.

```
library(fRLR)
set.seed(123)
n = 500
X = matrix(rnorm(10*n), 10, n)
Y = rnorm(10)
COV = matrix(rnorm(40), 10, 4)

#idx1 = c(1, 2, 3, 4, 1, 1, 1, 2, 2, 3)
#idx2 = c(2, 3, 4, 5, 3, 4, 5, 4, 5, 5)
id = combn(n, 2)
idx1 = id[1, ]
idx2 = id[2, ]

system.time(frlr2(X, idx1, idx2, Y, COV))

simpleLoop <- function()
{
  res = matrix(nrow=0, ncol=4)
  for (i in 1:length(idx1))
  {
    mat = cbind(X[, idx1[i]], X[,idx2[i]], COV)
    df = as.data.frame(mat)
    model = lm(Y~., data = df)
    tmp = c(idx1[i], idx2[i], summary(model)$coefficients[2,4], summary(model)$coefficients[3,4])
    res = rbind(res, tmp)
  }
}

system.time(simpleLoop())
```

The results are as follows.

![](https://stats.hohoweiya.xyz/images//frlr_speed.png)

As you can see, `fRLR` can speed this task significantly.

## Install Instructions

You'd better install this package on Linux or Mac on which perfectly support the GNU scientific library. Of course, you can still this package on windows if you setup your GSL environment, and I opened a repository [GSL](https://github.com/szcf-weiya/GSLwin) where you can find solutions to install GSL on windows.
