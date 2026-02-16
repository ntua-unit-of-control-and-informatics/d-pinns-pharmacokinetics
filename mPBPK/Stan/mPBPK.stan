functions{
//function containing the ODEs
real [] mPBPK(real t,
             real[] y,
             real[] theta,
             real[] rdata,
             int[] idata) {
  
  real dydt[4] ;
  real k21;real k12;real ke;
  real Vp; real VISF; real L; 
  real Vlymph; real Kp; real sigmaL; 
  real L1; real L2; real Vleaky; real Vtight;
  real dose_per_kg; real dose; real T_inf; real input; 
  real Hct; real f_ISF; real f_lymph;
  real sigma1; real sigma2; real CLp; real BW;
  
  sigma1 = theta[1];
  sigma2 = theta[2];
  CLp =theta[3];
  BW = theta[4];
  
  Hct = rdata[1];f_ISF = rdata[2];L = rdata[3];
  f_lymph = rdata[4];sigmaL = rdata[5];Kp = rdata[6];
  dose_per_kg = rdata[7]; T_inf = rdata[8]; 
  
  Vlymph = f_lymph*BW;
  Vp = Vlymph*(1-Hct);
  VISF = f_ISF*BW;
  dose = dose_per_kg * BW;
  L1 = 0.33*L;
  L2 = 0.67*L;
  Vtight = 0.65*VISF*Kp;
  Vleaky = 0.35*VISF*Kp;
  
 
  
  if (t<=T_inf){
    input = dose/T_inf;
  }else{
    input = 0;
  }
  
  dydt[1] = (input+y[4]*L-y[1]*L1*(1-sigma1)-y[1]*L2*(1-sigma2)-CLp*y[1])/Vp;
  dydt[2] =  (y[1]*L1*(1-sigma1) - y[2]*L1*(1-sigmaL))/Vtight;
  dydt[3] = (y[1]*L2*(1-sigma2) - y[3]*L2*(1-sigmaL))/Vleaky;
  dydt[4] = (y[2]*L1*(1-sigmaL) + y[3]*L2*(1-sigmaL) - y[4]*L)/Vlymph;

  
  return dydt;       
  } 
}

data {
  int<lower=1> N_obs;           // number of sampling points
  int<lower=1> N_subj;          // number of virtual individuals
  int<lower=1> N_compart;       // number of compartments of PBPK model
  real  params[8];      // Matrix containing the individual parameters
  
  real<lower=0> time[N_obs];
  real<lower=0> y0[N_compart];
  real t_init;
  
  vector[N_obs] mean_Cplasma;   // observed mean
  vector[N_obs] sd_Cplasma;     // observed sd
  
  // hyperpriors for CL distribution on normal scale
  real mean_mu_CL_normal;
  real sd_mu_CL_normal;
  real mean_tau_CL_normal;
  real sd_tau_CL_normal;
  
  vector[N_subj] BW;

  real rel_tol;
  real abs_tol;
  int  max_num_steps;
}

transformed data {
                
  // hyperpriors for CL distribution on log scale
  real mean_mu_CL;
  real sd_mu_CL;
  real mean_tau_CL;
  real sd_tau_CL;
  
  sd_mu_CL = sqrt(log(((sd_mu_CL_normal^2)/(mean_mu_CL_normal)^2)+1));
  mean_mu_CL= log(((mean_mu_CL_normal)^2)/sqrt((sd_mu_CL_normal^2)+(mean_mu_CL_normal)^2));
  sd_tau_CL= sqrt(log(((sd_tau_CL_normal^2)/(mean_tau_CL_normal)^2)+1));
  mean_tau_CL = log(((mean_tau_CL_normal)^2)/sqrt((sd_tau_CL_normal^2)+(mean_tau_CL_normal)^2));
       
}

parameters {
  real<lower=0> sigma_mean;
  real<lower=0> sigma_sd;
  
  real sigma1_raw;
  real sigma2_raw;
  
  real mu_CL;             
  real<lower=0> tau_CL;  
  vector[N_subj] z_CL;         
}

transformed parameters {
  real<lower=0, upper=1> sigma1;
  real<lower=0, upper=1> sigma2;
  vector<lower=0>[N_subj] CL;
  
  array[N_subj] matrix[N_obs, N_compart] y_hat;
  vector[N_obs] y_hat_mean;
  vector[N_obs] y_hat_sd;
  
  sigma1 = inv_logit(sigma1_raw);
  sigma2 = sigma1 * inv_logit(sigma2_raw);
  
  for (j in 1:N_subj) {
    vector[4] theta_tr;
    
    CL = exp(mu_CL + tau_CL * z_CL); 
    theta_tr[1] = sigma1;
    theta_tr[2] = sigma2;
    theta_tr[3] = CL[j];
    theta_tr[4] = BW[j];

      y_hat[j] =  to_matrix(integrate_ode_bdf(
                                    mPBPK,
                                    y0,
                                    t_init,
                                    time,
                                    to_array_1d(theta_tr),
                                    params,
                                    rep_array(0, 0),
                                    rel_tol,
                                    abs_tol,
                                    max_num_steps
                                   ));
  }
  
  for (i in 1:N_obs) {
    vector[N_subj] tmp;
    for (j in 1:N_subj){
      tmp[j] = y_hat[j][i, 1];
    }
    y_hat_mean[i] = mean(tmp);
    y_hat_sd[i]   = sd(tmp);
  }
}

model {
  // Priors on hyperparameters
  mu_CL  ~ normal(mean_mu_CL, sd_mu_CL);
  tau_CL ~ normal(mean_tau_CL, sd_tau_CL);
  z_CL ~ normal(0, 1);
   
  sigma_mean ~ normal(0, 1);
  sigma_sd   ~ normal(0, 1);
  sigma1_raw ~ normal(0, 1);
  sigma2_raw ~ normal(0, 1);
  
  // Summary-data likelihood on log-scale
  log(mean_Cplasma) ~ normal(log(fmax(y_hat_mean, 1e-12)), sigma_mean);
  log(sd_Cplasma)   ~ normal(log(fmax(y_hat_sd, 1e-12)),   sigma_sd);
}
generated quantities{
  real mu_CL_normal;
  real tau_CL_normal;
  vector[N_obs] mean_Cplasma_rep;
  vector[N_obs] sd_Cplasma_rep;
  
  mu_CL_normal = exp(mu_CL +(tau_CL^2)/2);
  tau_CL_normal = sqrt((exp(tau_CL^2)-1)*exp(2*mu_CL+tau_CL^2));
  


  for (i in 1:N_obs) {
    mean_Cplasma_rep[i] =
      lognormal_rng(log(fmax(y_hat_mean[i], 1e-12)), sigma_mean);
    sd_Cplasma_rep[i] =
      lognormal_rng(log(fmax(y_hat_sd[i],   1e-12)), sigma_sd);
  }
  
}

