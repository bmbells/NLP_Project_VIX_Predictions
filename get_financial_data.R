require(quantmod)
setwd("C:\\Users\\dabel\\Documents\\Natural_Language_Processing_MPCS\\project")
getSymbols("^VIX",from="1994-01-03")
getSymbols("^TNX",from="1994-01-03")
vx <- VIX[,c(1,4)]
tnx <- TNX[,c(1,4)]
df <- cbind(vx, tnx)
df$vix_1d <- (as.numeric(df[,2]) - as.numeric(df[,1]))/as.numeric(df[,1])
df$tnx_1d <- (as.numeric(df[,4]) - as.numeric(df[,3]))/as.numeric(df[,3])
df$vix_5d <- (as.numeric(df[,2]) - as.numeric(lag(df[,1],5)))/as.numeric(df[,1])
df$tnx_5d <- (as.numeric(df[,4]) - as.numeric(lag(df[,3],5)))/as.numeric(df[,3])
write.csv(as.data.frame(df),"financial_data.csv", row.names = TRUE)
