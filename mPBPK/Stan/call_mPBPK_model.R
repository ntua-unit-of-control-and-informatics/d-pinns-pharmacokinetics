library(rstan)
library(dplyr)
library(ggplot2)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores(), stanc.allow_optimizations = TRUE,
        stanc.auto_format = TRUE
)

#load data
pk_data <- read.csv("preprocessed_mPBPK_data.csv")
time <- pk_data$TIME
mean_Cplasma <- pk_data$MEAN
sd_Cplasma <- pk_data$SD

Hct <- 0.43  # hematocrit
f_ISF <- 0.15  # interstitial fluid fraction of total body weight
L <- 0.125 #  lymph flow (L/h)
f_lymph <- 0.073  # lymph/blood volume fraction of total body weight
sigma_L <- 0.2  # Lymphatic capillary reflection coefficient
Kp <- 0.8  # Available ISF fraction (fixed)
dose <-  1 #mg/kg 
T_inf <- 1.5 #hours
set.seed(123)
N_subj <- 30
BW_vec <- runif(N_subj, min = 50, max = 75)

#############################################################################################

pk_dat<-list(
  params = c(Hct, f_ISF, L, f_lymph, sigma_L, Kp, dose, T_inf),
  N_compart = 4, #Number of compartments
  N_subj = N_subj,		 #Number of virtual individuals
  N_obs = length(time), #Total number of observations
  time = time,
  mean_Cplasma = mean_Cplasma,
  sd_Cplasma = sd_Cplasma,
  t_init = 0,             
  y0 = c(0,0,0, 0),
  mean_mu_CL_normal = 0.01,
   sd_mu_CL_normal = 0.1,
   mean_tau_CL_normal= 0.01,
   sd_tau_CL_normal= 0.1,
  BW = BW_vec,
  rel_tol = 1e-5,
  abs_tol = 1e-5,
  max_num_steps = 1e5
)

tic = proc.time()

fit <- stan(file = 'mPBPK.stan', data = pk_dat, 
            iter = 400, warmup=200, chains=4)
tac = proc.time()
print(tac-tic)
options(max.print=5.5E5) 

print(fit, pars = c("sigma_mean", "sigma_sd", "sigma1", "sigma2",
                    "mu_CL", "tau_CL", #"CL", 
                     "mu_CL_normal", "tau_CL_normal"),
      digits = 4)

#Check diagnostic tools
check_hmc_diagnostics(fit)

#Use shinystan for inference
#library(shinystan)
#launch_shinystan(fit)

post <- rstan::extract(fit)

# mean_Cplasma_rep: iterations x N_obs
mean_rep <- post$mean_Cplasma_rep

## 2. Compute pointwise credible intervals over time
# e.g. 5%, 50%, 95% quantiles
mean_rep_q <- apply(mean_rep, 2, quantile, probs = c(0.025, 0.5, 0.975))

ppc_df <- data.frame(
  TIME = pk_data$TIME,
  q025  = mean_rep_q[1, ],
  q50  = mean_rep_q[2, ],
  q975  = mean_rep_q[3, ],
  MEAN_data = pk_data$MEAN,
  SD_data   = pk_data$SD
)

# For error bars: MEAN ± SD (truncate at 0 if needed)
ppc_df <- ppc_df %>%
  mutate(
    ymin_obs = pmax(MEAN_data - SD_data, 0),
    ymax_obs = MEAN_data + SD_data
  )

write.csv(ppc_df, "stan_posterior_predictive_check.csv")
## 3. Plot PPC band + observed mean with error bars
ggplot(ppc_df, aes(x = TIME)) +
  # posterior predictive 90% credible band
  geom_ribbon(aes(ymin = q025, ymax = q975),
              alpha = 0.2) +
  # posterior predictive median
  geom_line(aes(y = q50), linewidth = 1) +
  # observed means with error bars
  geom_errorbar(aes(ymin = ymin_obs, ymax = ymax_obs),
                width = 0.1) +
  geom_point(aes(y = MEAN_data), size = 2) +
  labs(
    x = "Time (h)",
    y = "Plasma concentration",
    title = "Posterior predictive check (mean concentrations)"
  ) +
  theme_bw()
