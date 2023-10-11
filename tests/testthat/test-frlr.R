tol = 1e-10
set.seed(123)
X = matrix(rnorm(50), 10, 5)
Y = rnorm(10)
COV = matrix(rnorm(40), 10, 4)
res.frlr = frlr1(X, Y, COV, 1)

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

test_that("frlr1 vs lm", {
  expect_lt(
    max(abs(res.frlr[,2] - res[,2])),
    tol)
})

### frlr2
idx1 = c(1, 2)#, 3, 4, 1, 1, 1, 2, 2, 3)
idx2 = c(2, 3)#, 4, 5, 3, 4, 5, 4, 5, 5)

res.frlr2 = frlr2(X, idx1, idx2, Y, COV, 1)

res = matrix(nrow=0, ncol=4)
for (i in 1:length(idx1))
{
  mat = cbind(X[, idx1[i]], X[,idx2[i]], COV)
  df = as.data.frame(mat)
  model = lm(Y~., data = df)
  tmp = c(idx1[i], idx2[i], summary(model)$coefficients[2,4], summary(model)$coefficients[3,4])
  res = rbind(res, tmp)
}

test_that("frlr2 vs lm", {
  expect_lt(
    max(abs(res.frlr2[,3] - res[,3])),
    tol)
  expect_lt(
    max(abs(res.frlr2[,4] - res[,4])),
    tol)
})