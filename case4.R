# https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise

library(e1071)
library(readxl)

CVCVSVR<-function(x_train,y_train,x_test,y_test){
  
  t1<-Sys.time()
  
  yourdata<-cbind(x_train,y_train)
  
  yourdata<-yourdata[sample(nrow(yourdata)),]
  
  folds <- cut(seq(1,nrow(yourdata)),breaks=10,labels=FALSE)
  
  cvPerformance<-array()
  
  j<-0
  
  ChosenEps<-c(0.01, 0.05, 0.1, 0.2 ,0.3)
  
  for (Select_Eps in ChosenEps){
    
    Calculate_subRMSE<-array()
    
    j<-j+1
    
    for(i in 1:10){
      
      testIndexes <- which(folds==i,arr.ind=TRUE)
      
      testData <- yourdata[testIndexes, ]
      
      trainData <- yourdata[-testIndexes, ]
      
      trainDataX<-trainData[,1:5]
      
      trainDataY<-trainData[,6]
      
      testDataX<-testData[,1:5]
      
      testDataY<-testData[,6]
      
      subcvsvm<-svm(trainDataX,trainDataY,type='eps-regression',kernel="radial",epsilon=Select_Eps)
      
      subcvPrediction<-predict(subcvsvm,testDataX)
      
      subRMSE<-sqrt(mean((subcvPrediction-testDataY)^2))
      
      Calculate_subRMSE[i]<- subRMSE
      
    }
    
    cvPerformance[j]<-mean(Calculate_subRMSE)
    
  }
  
  CVEps<-ChosenEps[order(cvPerformance)[1]]
  
  cvsvm<-svm(x_train,y_train,type='eps-regression',kernel="radial",epsilon=CVEps)
  
  cvPrediction<-predict(cvsvm,x_test)
  
  cvRMSE<-sqrt(mean((cvPrediction-y_test)^2))
  
  cvMAE<-mean(abs(cvPrediction-y_test))
  
  t2<-Sys.time()
  
  cv_time<-t2-t1
  
  return(list(cvRMSE=cvRMSE,cvMAE=cvMAE, cv_time=cv_time))
  
}

DataSet<-read.table('airfoil_self_noise.dat')

ModelPerformance<-array()

for (i in c(1:100)){
  
  train_sub<-sample(nrow(DataSet),7/10*nrow(DataSet))
  
  x_train<-scale(DataSet[train_sub,1:5],scale=TRUE)
  
  x_centre<-apply(DataSet[train_sub,1:5],2,mean)
  
  x_sd<-apply(DataSet[train_sub,1:5],2,sd)
  
  y_train<-DataSet[train_sub,6]
  
  x_test1<-sweep(DataSet[-train_sub,1:5],2,x_centre,"-")
  
  x_test<-sweep(x_test1,2,x_sd,"/")
  
  y_test<-DataSet[-train_sub,6]
  
  t1<-Sys.time()
  
  GeneralSVR<-svm(x_train,y_train,type='eps-regression',kernel="radial")
  
  res<-GeneralSVR$residuals
  
  epiloss<-function(Para)
    
    # Para<-array(ep,sigma)  
    
    # ep<Para[1]
    
    # sigma<-Para[2]  
    
  {
    Llog<-ifelse(abs(res/Para[2])<Para[1],0,abs(res/Para[2] )-Para[1])
    
    sum(log(Para[2])+log(2*(1+Para[1]))+Llog)
    
  }
  
  optimun<-optim(c(10,2),epiloss,method="L-BFGS-B",lower=c(0.00001,0.01))$par
  
  RobustEps<-optimun[1]
  
  Scale<-optimun[2]
  
  scale_x_train<-x_train
  
  scale_y_train<-y_train/Scale
  
  scale_x_test<-x_test
  
  C2<-quantile(abs(scale_y_train),.95)
  
  OptimalSVR<-svm(scale_x_train,scale_y_train,type='eps-regression',kernel="radial",scale=FALSE,epsilon=RobustEps, cost=C2)
  
  scale_prediction<-predict(OptimalSVR,scale_x_test)
  
  Prediction<-scale_prediction*Scale
  
  RobustSVREstimation<-list(TestOutput=Prediction,RobustEpsilon=RobustEps,ScaleEstimation=Scale, CEstimation=C2)
  
  RRMSE<-sqrt(mean((Prediction-y_test)^2))
  
  RMAE<-mean(abs(Prediction-y_test))
  
  t2<-Sys.time()
  
  RobustT<-t2-t1
  # default SVR
  
  ModelGeneral<-svm(x_train,y_train, type='eps-regression', kernel = "radial")
  
  GPrediction<-predict(ModelGeneral,x_test)
  
  GRMSE<-sqrt(mean((GPrediction-y_test)^2))
  
  GMAE<-mean(abs(GPrediction-y_test))
  
  
  ## CV SVR 
  
  CVResults<-CVCVSVR(x_train,y_train,x_test,y_test)
  
  CVRMSE<-CVResults$cvRMSE
  
  CVMAE<-CVResults$cvMAE
  
  CVT<-CVResults$cv_time
  
  ## Cherkassky, Ma 2004
  
  ModelGeneral1<-svm(x_train,y_train, type='eps-regression', kernel = "radial")
  
  Mean_y_train<-mean(y_train)
  
  Sd_y_train<-sd(y_train)
  
  scale_residual<-(ModelGeneral1$residuals-Mean_y_train)/Sd_y_train
  
  NumObs<-length(y_train)
  
  MaEps<-3*sd(scale_residual)*sqrt(log(NumObs)/NumObs)
  
  MaSVR<-svm(x_train,y_train, type='eps-regression', kernel = "radial", epsilon=MaEps)
  
  MaPrediction<-predict(MaSVR,x_test)
  
  MaRMSE<-sqrt(mean((MaPrediction-y_test)^2))
  
  MaMAE<-mean(abs(MaPrediction-y_test))
  
  A<-c(Scale,RobustEps, GMAE, MaMAE, CVMAE, RMAE,  GRMSE,  MaRMSE, CVRMSE, RRMSE)
  
  print(A)
  
  ModelPerformance<-rbind(ModelPerformance,A)
  
}


Results<-apply(ModelPerformance[2:101,],2,mean)

print(Results)

ModelPerformance<-rbind(ModelPerformance,Results)

write.csv(ModelPerformance,"newcase4.csv")
