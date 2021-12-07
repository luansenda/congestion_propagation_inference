library(dbnR)
library(bnlearn)
library(openxlsx)

sp <- read.csv("C:\\Users\\g\\Desktop\\anomaly tree\\speed_data\\net_2.csv")

## -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# structure learning -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

dt_train <- sp[0:540,]
dt_val <- sp[541:1080,]

size = 2
## constrains
white_list <- rbind(c("R4_t_0", "R5_t_0"),c("R5_t_0", "R6_t_0"),c("R1_t_0", "R2_t_0"))
black_list <- rbind(c("R12_t_0", "R11_t_0"))
net <- learn_dbn_struc(dt_train, size, method = "dmmhc", whitelist = white_list, blacklist = black_list)

## no constrains
net <- learn_dbn_struc(dt_train, size, method = "dmmhc")
plot_dynamic_network(net)

#net <- learn_dbn_struc(dt_train, size, method = "dmmhc", whitelist = whitelist, blacklist = blacklist,
#                       restrict = "mmpc", maximize = "hc", 
#                       restrict.args = list(test = "cor"),
#                       maximize.args = list(score = "bic-g", maxp = 3))

## -#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-# Parameters learning -#-#-#-#-#-#-#-#-#-#-#-#-
f_dt_train <- fold_dt(dt_train, size)
fit <- fit_dbn_params(net, f_dt_train, method = "mle")


## -------- Conditional density save ---------------------
weigh_num = 0
for (i in 1:length(fit)) {
  if(length(fit[[i]]$coefficients)>1){
    weigh_num = weigh_num + length(fit[[i]]$coefficients)-1
  }
}

wei_arr = array(0,dim = c(weigh_num,3))# initialize weight array
raw_ind = 0
for (i in 1:length(fit)) {
  if(length(fit[[i]]$coefficients)>1){
    for (j in 2:length(fit[[i]]$coefficients)) {
      raw_ind = raw_ind+1
      wei_arr[raw_ind,1]=names(fit[[i]]$coefficients[j])
      wei_arr[raw_ind,2]=names(fit[i])
      wei_arr[raw_ind,3]=round(fit[[i]]$coefficients[j][[1]],2)
    }
  }
}
write.csv(wei_arr,file = "C:/Users/g/Desktop/anomaly tree/DBN_RES/net2_weigh.csv",row.names = FALSE)