require(OpenMx)
source('Heritability_model_function.R')

# Load Data
# data <- read.csv('data/separate_pca/story_rgspec_vx100.csv')

# roiname_all <- c('L_AG', 'L_MTG', 'L_ATL', 'L_IFG',
#                  'R_AG', 'R_MTG', 'R_ATL', 'R_IFG')

# Load Data
# data <- read.csv('data/separate_pca/math_rgspec_vx100.csv')

# roiname_all <- c('L_aIPS', 'L_dIPS', 'L_SPL', 'L_MTL', 'L_latSMA', 'L_MFG', 'L_INS', 'L_medSMA', 'L_vlPFC',
#                  'R_aIPS', 'R_dIPS', 'R_SPL', 'R_MTL', 'R_latSMA', 'R_MFG', 'R_INS', 'R_medSMA', 'R_vlPFC')

# Load Data
# data <- read.csv('pre-heritability_TMA.csv')
# roiname_all <- c('IOG_thickness_lh', 'pFus_thickness_lh', 'mFus_thickness_lh', 'pFus_mFus_thickness_lh', 'IOG_thickness_rh', 'pFus_thickness_rh', 'mFus_thickness_rh', 'pFus_mFus_thickness_rh', 'IOG_myelin_lh', 'pFus_myelin_lh', 'mFus_myelin_lh', 'pFus_mFus_myelin_lh', 'IOG_myelin_rh', 'pFus_myelin_rh', 'mFus_myelin_rh', 'pFus_mFus_myelin_rh', 'IOG_activ_lh', 'pFus_activ_lh', 'mFus_activ_lh', 'pFus_mFus_activ_lh', 'IOG_activ_rh', 'pFus_activ_rh', 'mFus_activ_rh', 'pFus_mFus_activ_rh')

data <- read.csv('pre-heritability_rsfc.csv')
roiname_all <- c('IOG_trg1_lh', 'pFus_trg1_lh', 'mFus_trg1_lh', 'pFus_mFus_trg1_lh', 'IOG_trg2_lh', 'pFus_trg2_lh', 'mFus_trg2_lh', 'pFus_mFus_trg2_lh', 'IOG_trg3_lh', 'pFus_trg3_lh', 'mFus_trg3_lh', 'pFus_mFus_trg3_lh', 'IOG_trg4_lh', 'pFus_trg4_lh', 'mFus_trg4_lh', 'pFus_mFus_trg4_lh', 'IOG_trg5_lh', 'pFus_trg5_lh', 'mFus_trg5_lh', 'pFus_mFus_trg5_lh', 'IOG_trg6_lh', 'pFus_trg6_lh', 'mFus_trg6_lh', 'pFus_mFus_trg6_lh', 'IOG_trg7_lh', 'pFus_trg7_lh', 'mFus_trg7_lh', 'pFus_mFus_trg7_lh', 'IOG_trg8_lh', 'pFus_trg8_lh', 'mFus_trg8_lh', 'pFus_mFus_trg8_lh', 'IOG_trg9_lh', 'pFus_trg9_lh', 'mFus_trg9_lh', 'pFus_mFus_trg9_lh', 'IOG_trg10_lh', 'pFus_trg10_lh', 'mFus_trg10_lh', 'pFus_mFus_trg10_lh', 'IOG_trg11_lh', 'pFus_trg11_lh', 'mFus_trg11_lh', 'pFus_mFus_trg11_lh', 'IOG_trg12_lh', 'pFus_trg12_lh', 'mFus_trg12_lh', 'pFus_mFus_trg12_lh', 'IOG_trg1_rh', 'pFus_trg1_rh', 'mFus_trg1_rh', 'pFus_mFus_trg1_rh', 'IOG_trg2_rh', 'pFus_trg2_rh', 'mFus_trg2_rh', 'pFus_mFus_trg2_rh', 'IOG_trg3_rh', 'pFus_trg3_rh', 'mFus_trg3_rh', 'pFus_mFus_trg3_rh', 'IOG_trg4_rh', 'pFus_trg4_rh', 'mFus_trg4_rh', 'pFus_mFus_trg4_rh', 'IOG_trg5_rh', 'pFus_trg5_rh', 'mFus_trg5_rh', 'pFus_mFus_trg5_rh', 'IOG_trg6_rh', 'pFus_trg6_rh', 'mFus_trg6_rh', 'pFus_mFus_trg6_rh', 'IOG_trg7_rh', 'pFus_trg7_rh', 'mFus_trg7_rh', 'pFus_mFus_trg7_rh', 'IOG_trg8_rh', 'pFus_trg8_rh', 'mFus_trg8_rh', 'pFus_mFus_trg8_rh', 'IOG_trg9_rh', 'pFus_trg9_rh', 'mFus_trg9_rh', 'pFus_mFus_trg9_rh', 'IOG_trg10_rh', 'pFus_trg10_rh', 'mFus_trg10_rh', 'pFus_mFus_trg10_rh', 'IOG_trg11_rh', 'pFus_trg11_rh', 'mFus_trg11_rh', 'pFus_mFus_trg11_rh', 'IOG_trg12_rh', 'pFus_trg12_rh', 'mFus_trg12_rh', 'pFus_mFus_trg12_rh')

ACE_h2estimate_matric <- matrix(1, 3, length(roiname_all))
ACE_c2estimate_matric <- matrix(1, 3, length(roiname_all))
ACE_e2estimate_matric <- matrix(1, 3, length(roiname_all))

AE_h2estimate_matric <- matrix(1, 3, length(roiname_all))
AE_e2estimate_matric <- matrix(1, 3, length(roiname_all))
for (i in seq_along(roiname_all))
{
  # Select Variables for Analysis
  selVars <- c(paste(roiname_all[i],'1',sep=''), paste(roiname_all[i],'2',sep=''))
  aceVars   <- c("A1","C1","E1","A2","C2","E2")
  
  # Select Data for Analysis
  mzData    <- subset(data, zyg==1, selVars)
  dzData    <- subset(data, zyg==3, selVars)
  
  AEmodelFit <- Twin_AEmodel(mzData, dzData, selVars)
  ACEmodelFit <- Twin_ACEmodel(mzData, dzData, selVars)
  print(ComparedModel <- ModelComparison(ACEmodelFit, AEmodelFit))
  AE_confint_tmp <- summary(AEmodelFit)$CI
  ACE_confint_tmp <- summary(ACEmodelFit)$CI
  
  ACE_h2estimate_matric[1:3, i] <- cbind(ACE_confint_tmp[1,1], ACE_confint_tmp[1,2], ACE_confint_tmp[1,3])
  ACE_c2estimate_matric[1:3, i] <- cbind(ACE_confint_tmp[2,1], ACE_confint_tmp[2,2], ACE_confint_tmp[2,3])
  ACE_e2estimate_matric[1:3, i] <- cbind(ACE_confint_tmp[3,1], ACE_confint_tmp[3,2], ACE_confint_tmp[3,3])
  
  AE_h2estimate_matric[1:3, i] <- cbind(AE_confint_tmp[1,1], AE_confint_tmp[1,2], AE_confint_tmp[1,3])
  AE_e2estimate_matric[1:3, i] <- cbind(AE_confint_tmp[3,1], AE_confint_tmp[3,2], AE_confint_tmp[3,3])
  
  # DataFrame
  ACE_h2estimate <- as.data.frame(ACE_h2estimate_matric)
  names(ACE_h2estimate) <- roiname_all
  ACE_c2estimate <- as.data.frame(ACE_c2estimate_matric)
  names(ACE_c2estimate) <- roiname_all
  ACE_e2estimate <- as.data.frame(ACE_e2estimate_matric)
  names(ACE_e2estimate) <- roiname_all
  AE_h2estimate <- as.data.frame(AE_h2estimate_matric)
  names(AE_h2estimate) <- roiname_all
  AE_e2estimate <- as.data.frame(AE_e2estimate_matric)
  names(AE_e2estimate) <- roiname_all
}




