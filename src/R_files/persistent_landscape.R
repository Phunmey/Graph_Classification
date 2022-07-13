library(TDA)

X <- circleUnif(50)+rnorm(50,sd=0.05) 
plot(X,asp=1) 
maxScale <- 2 
Diag <- ripsDiag(X,maxdimension = 1,maxscale = maxScale)$diagram  
#persistence landscape 
tseq <- seq(0,maxScale,length = 10) 
Land <- landscape(Diag, dimension = 1, KK =1, tseq)
silh <- silhouette(Diag,p=2,dimension = 1,tseq)

par(mfrow = c(1,3))
plot.diagram(Diag)
plot(tseq, Land, type = "l", xlab = "t", ylab = "landscape", asp = 1)
plot(tseq, silh, type = "l", xlab = "t", ylab = "silhouette", asp = 1)
par(mfrow = c(1,1))
# 
# Diag <- matrix(c(0, 0, 10, 1, 0, 3, 1, 3, 8), ncol = 3, byrow = TRUE)
# DiagLim <- 10
# colnames(Diag) <- c("dimension", "Birth", "Death")
# 
# #persistence landscape
# tseq <- seq(0,DiagLim, length = 1000)
# Land <- landscape(Diag, dimension = 1, KK = 1, tseq)
# 
# par(mfrow = c(1,2))
# plot.diagram(Diag)
# plot(tseq, Land, type = "l", xlab = "t", ylab = "landscape", asp = 1)