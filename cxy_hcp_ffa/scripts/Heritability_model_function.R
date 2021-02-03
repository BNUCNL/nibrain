require(OpenMx)

Twin_AEmodel <- function(mzData, dzData, roiname) 
{
  # Select Variables for Analysis
  # roiname: c(roiname1, roiname2)
  selVars   <- roiname
  aceVars   <- c("A1","C1","E1","A2","C2","E2")
  
  # Data objects for Multiple Groups
  dataMZ       <- mxData( observed=mzData, type="raw" )
  dataDZ       <- mxData( observed=dzData, type="raw" )

  # variances of latent variables
  latVariances <- mxPath( from=aceVars, arrows=2, 
                          free=FALSE, values=1 )
  # means of latent variables
  latMeans     <- mxPath( from="one", to=aceVars, arrows=1, 
                          free=FALSE, values=0 )
  # means of observed variables
  obsMeans     <- mxPath( from="one", to=selVars, arrows=1, 
                          free=TRUE, values=3, labels="mean" )

  # covariance between C1 & C2
  covC1C2      <- mxPath( from="C1", to="C2", arrows=2, 
                          free=FALSE, values=1 )
  
  # covariance between A1 & A2 in MZ twins
  covA1A2_MZ   <- mxPath( from="A1", to="A2", arrows=2, 
                          free=FALSE, values=1 )
  # covariance between A1 & A2 in DZ twins
  covA1A2_DZ   <- mxPath( from="A1", to="A2", arrows=2, 
                          free=FALSE, values=.5 )
  
  pathAeT1    <- mxPath( from=c("A1","C1","E1"), to=selVars[1], arrows=1, 
                         free=c(T,F,T), values=c(1,0,1),  label=c("a_AE","c_AE","e_AE") )
  # path coefficients for twin 2
  pathAeT2    <- mxPath( from=c("A2","C2","E2"), to=selVars[2], arrows=1, 
                         free=c(T,F,T), values=c(1,0,1),  label=c("a_AE","c_AE","e_AE") )
  # Combine Groups
  paths_AE        <- list( latVariances, latMeans, obsMeans,
                           pathAeT1, pathAeT2, covC1C2 )
  modelMZ_AE      <- mxModel(model="MZ", type="RAM", manifestVars=selVars, 
                             latentVars=aceVars, paths_AE, covA1A2_MZ, dataMZ )
  modelDZ_AE      <- mxModel(model="DZ", type="RAM", manifestVars=selVars,
                             latentVars=aceVars, paths_AE, covA1A2_DZ, dataDZ )
  minus2ll     <- mxAlgebra( expression=MZ.fitfunction + DZ.fitfunction, 
                             name="minus2loglikelihood" )
  obj          <- mxFitFunctionAlgebra( "minus2loglikelihood" )
  
  A_AE            <- mxAlgebra(a_AE %*% t(a_AE), name='A_AE')
  C_AE            <- mxAlgebra(c_AE %*% t(c_AE), name='C_AE')
  E_AE            <- mxAlgebra(e_AE %*% t(e_AE), name='E_AE')
  V_AE            <- mxAlgebra(A_AE+C_AE+E_AE, name='V_AE')
  
  h2_AE           <- mxAlgebra(A_AE/V_AE, name='h2_AE')
  c2_AE           <- mxAlgebra(C_AE/V_AE, name='c2_AE')
  e2_AE           <- mxAlgebra(E_AE/V_AE, name='e2_AE')
  ci_AE           <- mxCI(c('AE.h2_AE', 'AE.c2_AE', 'AE.e2_AE'))
  
  twinAEModel     <- mxModel(model="AE", modelMZ_AE, modelDZ_AE, A_AE, C_AE, E_AE, V_AE, h2_AE, c2_AE, e2_AE, ci_AE, minus2ll, obj )
  # Run Model
  twinAEFit   <- mxRun(twinAEModel, intervals = TRUE)
  return(twinAEFit)
}

Twin_ACEmodel <- function(mzData, dzData, roiname) 
{
  # Select Variables for Analysis
  selVars   <- roiname
  aceVars   <- c("A1","C1","E1","A2","C2","E2")
  
  # Data objects for Multiple Groups
  dataMZ       <- mxData( observed=mzData, type="raw" )
  dataDZ       <- mxData( observed=dzData, type="raw" )
  
  # variances of latent variables
  latVariances <- mxPath( from=aceVars, arrows=2, 
                          free=FALSE, values=1 )
  # means of latent variables
  latMeans     <- mxPath( from="one", to=aceVars, arrows=1, 
                          free=FALSE, values=0 )
  # means of observed variables
  obsMeans     <- mxPath( from="one", to=selVars, arrows=1, 
                          free=TRUE, values=3, labels="mean" )
  # path coefficients for twin 1
  pathAceT1    <- mxPath( from=c("A1","C1","E1"), to=selVars[1], arrows=1, 
                          free=c(T,T,T), values=c(1,1,1),  label=c("a_ACE","c_ACE","e_ACE") )
  # path coefficients for twin 2
  pathAceT2    <- mxPath( from=c("A2","C2","E2"), to=selVars[2], arrows=1, 
                          free=c(T,T,T), values=c(1,1,1),  label=c("a_ACE","c_ACE","e_ACE") )
  # covariance between C1 & C2
  covC1C2      <- mxPath( from="C1", to="C2", arrows=2, 
                          free=FALSE, values=1 )
  
  # covariance between A1 & A2 in MZ twins
  covA1A2_MZ   <- mxPath( from="A1", to="A2", arrows=2, 
                          free=FALSE, values=1 )
  # covariance between A1 & A2 in DZ twins
  covA1A2_DZ   <- mxPath( from="A1", to="A2", arrows=2, 
                          free=FALSE, values=.5 )
  # Combine Groups
  paths_ACE        <- list( latVariances, latMeans, obsMeans,
                            pathAceT1, pathAceT2, covC1C2 )
  modelMZ_ACE      <- mxModel(model="MZ", type="RAM", manifestVars=selVars, 
                              latentVars=aceVars, paths_ACE, covA1A2_MZ, dataMZ )
  modelDZ_ACE      <- mxModel(model="DZ", type="RAM", manifestVars=selVars, 
                              latentVars=aceVars, paths_ACE, covA1A2_DZ, dataDZ )
  
  A_ACE            <- mxAlgebra(a_ACE %*% t(a_ACE), name='A_ACE')
  C_ACE            <- mxAlgebra(c_ACE %*% t(c_ACE), name='C_ACE')
  E_ACE            <- mxAlgebra(e_ACE %*% t(e_ACE), name='E_ACE')
  V_ACE            <- mxAlgebra(A_ACE+C_ACE+E_ACE, name='V_ACE')
  
  h2_ACE           <- mxAlgebra(A_ACE/V_ACE, name='h2_ACE')
  c2_ACE           <- mxAlgebra(C_ACE/V_ACE, name='c2_ACE')
  e2_ACE           <- mxAlgebra(E_ACE/V_ACE, name='e2_ACE')
  ci_ACE           <- mxCI(c('ACE.h2_ACE', 'ACE.c2_ACE', 'ACE.e2_ACE'))
  minus2ll     <- mxAlgebra( expression=MZ.fitfunction + DZ.fitfunction, 
                             name="minus2loglikelihood" )
  obj          <- mxFitFunctionAlgebra( "minus2loglikelihood" )
  
  twinACEModel     <- mxModel(model="ACE", modelMZ_ACE, modelDZ_ACE, A_ACE, C_ACE, E_ACE, V_ACE, h2_ACE, c2_ACE, e2_ACE, ci_ACE, minus2ll, obj )
  
  # Run Model
  twinACEFit   <- mxRun(twinACEModel, intervals = TRUE)
  return(twinACEFit)
}

ModelComparison <- function(Model1Fit, Model2Fit)
{
    ComparedModel <- rbind(mxCompare(Model1Fit, Model2Fit))
    return(ComparedModel)
}

SwapTableColumns <- function(data_frame, column_pair1, column_pair2)
{
    data_tmp <- data_frame
    data_length <- nrow(data_frame)

    swap_indicator <- sample(c(TRUE, FALSE), data_length, replace=TRUE)
    for (i in seq(data_length))
    {
        if (swap_indicator[i])
        {
          # Swap each row
          swap_variable <- data_tmp[i, column_pair1]
          data_tmp[i, column_pair1] <- data_tmp[i, column_pair2]
          data_tmp[i, column_pair2] <- swap_variable
        }
    }
    return(data_tmp)
}
