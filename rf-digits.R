zz=gzfile('zip.train.gz','rt')   
x.train=read.csv(zz,header=F,sep=" ")
x.train <- x.train[,1:257]
zz=gzfile('zip.test.gz','rt')   
x.test=read.csv(zz,header=F,sep=" ")
x.test <- x.test[,1:257]

mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

## RANDOM FOREST
library(randomForest)
y.tr <- as.factor(x.train$V1)
rf <- randomForest(as.factor(V1) ~ ., data = x.train, ntree = 50)
rf.pr <- predict(rf, newdata=x.test, type='class')
rf.pr.acc <- 1-sum(as.numeric(as.numeric(levels(rf.pr))[rf.pr] - as.numeric(x.test$V1) != 0))/dim(x.test)[1]

# NORMAL TREE
library(rpart)
tree <- rpart(as.factor(V1) ~ ., data = x.train, method='class')
plot(tree, uniform=FALSE, margin=0.1)
text(tree,use.n=F)

tree.pr <- predict(tree, newdata=x.test, type='class')
tree.pr.acc <- 1- sum(as.numeric(as.numeric(levels(tree.pr))[tree.pr]- as.numeric(x.test$V1) != 0))/dim(x.test)[1]
tree.pr.acc

## BAGGING

my.c <- rpart.control(minsplit = 3, cp = 1e-6, xval = 10)
NB <- 20
ts <- vector('list',NB)
n <- nrow(x.train)
for (j in 1:NB){
  print(j)
  ii <- sample(1:n, replace=TRUE)
  ts[[j]] <- rpart(V1 ~ ., data = x.train[ii,], method = 'class', parms = list(split = 'information'), control= my.c)
}

## majority vote
prs <- list()
for(j in 1:NB) {
  pr <- predict(ts[[j]], newdata=x.test, type='class')
  prs[[j]] <- as.matrix(pr)
}
prs.mx = do.call(cbind, prs)
prs.bagg <- c()
for (i in 1:dim(prs.mx)[1]){
  prs.bagg <- c(prs.bagg, mode(prs.mx[i,]))
}

as.numeric(prs.bagg) - as.numeric(x.test$V1)
bagg.pr.acc <- 1- sum(as.numeric(as.numeric(prs.bagg) - as.numeric(x.test$V1) != 0))/dim(x.test)[1]