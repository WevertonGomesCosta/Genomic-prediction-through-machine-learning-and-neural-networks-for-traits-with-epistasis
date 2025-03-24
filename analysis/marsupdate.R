pathFenT <- file.path("data/PHEN_T")

pathFenV <- file.path("data/PHEN_V")

pathGenT <- file.path("data/GEN_T")

pathGenV <- file.path("data/GEN_V")

nvariavel <- 18
nfolds <- length(pathFenT)
grau <- 3

library(earth)                     # fit MARS models
library(caret); library(vip)       # variable importance
library(tidyverse)                 # Manipula??o de Dados
library(writexl)

set.seed(123)                    #Semente para repodu??o dos resultados    
threshi <- 0.001
threshf <- 0.05
passot <- 0.005

newvari <- 0.01
newvarf <- 0.2
passon <- 0.05


for(g in 1:grau){
  
  for(j in 1:nvariavel) {
    
    cat("======================================================","\n")
    cat("Variavel: ",j,"\n")
    cat("======================================================","\n")
    
    R2t<-matrix(nrow=nfolds, ncol =1) #Objeto r? treinamento
    R2v<-matrix(nrow=nfolds, ncol =1) #Objeto r? valida??o
    reqt<-matrix(nrow=nfolds, ncol =1)#Objeto RQEM treinamento
    reqv<-matrix(nrow=nfolds, ncol =1)#Objeto RQEM valida??o
    
    for(i in 1:nfolds){
      
      dadosTy <-read.table(pathFenT[i])
      dadosTx <-read.table(pathGenT[i])
      dadosVy <-read.table(pathFenV[i])
      dadosVx <-read.table(pathGenV[i])
      # *******************************************************************
      # Fit a basic MARS model Cubico
      # *******************************************************************
      cat("K-fold = ",i, "\n")
      
      for(k in c(threshi ,seq(passot , threshf , by = passot))){
        for(n in c(newvari ,seq(passon , newvarf , by = passon))){
          
        mars3 <- earth(x=dadosTx,y=dadosTy[,j], degree=g, thresh = k, newvar.penalty = n) #Mudan?a em R? menor que 0.05, crit?rio de parada
        
        ## Predi??o do modelo c?bico
        ypred <- predict(mars3, dadosVx)
        
        ##Par?metros do Modelo
        rv <- cor(ypred,dadosVy[,j])
        R2vk <- rv*rv
        
        if(k == threshi & n == newvari){
          cat("======================================================","\n")
          cat("Resultado para Variavel: ",j," Grau: ",g,"Mudan?a em R?: ",k,"Mudan?a em penalidade: ",n,"\n")
          cat("======================================================","\n")
          R2v[i] <- R2vk
          best.mars<-mars3
          k1<-k
          n1<-n
        } 
        
        if(R2vk > R2v[i]){
          cat("======================================================","\n")
          cat("Resultado para Variavel: ",j," Grau: ",g,"Mudan?a em R?: ",k,"Mudan?a em penalidade: ",n,"\n")
          cat("======================================================","\n")
          R2v[i] <- R2vk
          best.mars<-mars3
          k1<-k
          n1<-n
        }
        
      }
      }
      
      cat("======================================================","\n")
      cat("Resultado para Variavel: ",j," Grau: ",g,"Mudan?a em R?: ",k1,"Mudan?a em penalidade: ",n1,"\n")
      cat("======================================================","\n")
      
      ## Par?metros do modelo Treinamento
      rt <- cor(best.mars$fitted.values,dadosTy[,j])
      R2t[i] <- rt*rt
      errot<-best.mars$fitted.values-dadosTy[,j]
      reqt[i] <- sqrt(mean(errot^2))
      
      ## Par?metros do modelo Valida??o
      errov<-dadosVy[,j]-ypred
      reqv[i] <- sqrt(mean(errov^2))
      coefc<- list(best.mars$coefficients)
      
      
      ## Import?ncia de marcadores
      imp<-evimp(best.mars, trim = FALSE)
      imp<-as.data.frame(unclass(imp[,c(1,6)]))
      names<-cbind(imp$col,i)
      imp<-data.frame(imp$rss,names)
      colnames(imp)<-c("Overall","marker", "n fold")
      
      if (i == 1){ 
        imp.mars3 <- imp
      } else {
        imp.mars3 <- imp.mars3 %>% rbind(imp)
      }
      
    }
    
    cat("Par?metros do modelo da variavel  ",j, "\n")
    par.mars3 <- cbind(R2t, R2v, reqt, reqv)
    par.mars3 <- rbind(par.mars3,apply(par.mars3,2,mean),apply(par.mars3,2,sd))
    colnames(par.mars3) <- c("R? Trein","R? Val",  "REQM Trein", "REQM Val")
    rownames(par.mars3) <- c("K-Fold 1","K-Fold 2",  "K-Fold 3", "K-Fold 4","K-Fold 5","Mean", "SD")
    
    if(g ==1 ){
      names<-cbind(rownames(par.mars3),"MARS L",j)
      namesi<-cbind("MARS L",rep(j,len = nrow(imp.mars3)))
    } 
    if(g == 2 ){
      names<-cbind(rownames(par.mars3),"MARS Q",j)
      namesi<-cbind("MARS Q",rep(j,len = nrow(imp.mars3)))
    } 
    if(g == 3 ){
      names<-cbind(rownames(par.mars3),"MARS C",j)
      namesi<-cbind("MARS C",rep(j,len = nrow(imp.mars3)))
    }
    
    colnames(names)<-c("n Fold", "method", "variable")
    par.mars3<-data.frame(par.mars3,names)
    par.mars3
    
    colnames(namesi)<-c("method", "variable")
    imp.mars3<-data.frame(imp.mars3,namesi)
    
    #Resultado de todas variaveis
    if (j == 1){ 
      res.mars3 <- par.mars3
      res.imp.mars3 <- imp.mars3
    } else {
      res.mars3 <- res.mars3 %>% rbind(par.mars3)
      res.imp.mars3 <- res.imp.mars3 %>% rbind(imp.mars3)
    }
    
write_xlsx(res.mars3,  paste("res.mars",j,g,".xlsx", sep = ""))

arq <- paste("res.imp.mars",j,g,".Rdata", sep = "")
save(res.imp.mars3,  file = arq)
    
  }
  
  
  cat("Resultado final de todas variav?is para MARS",g, "\n")
  write_xlsx(res.mars3,  paste("res.mars",g,".xlsx", sep = ""))
  
  cat("Import?ncia de marcadores para todas variav?is do MARS", g, "\n")
  arq<-paste("res.imp.mars",g,".RData", sep = "")
  save(res.imp.mars3,  file = arq)
  
}
