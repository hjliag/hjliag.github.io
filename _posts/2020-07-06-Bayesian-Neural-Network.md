---
title:  "Bayesian Neural Network"
---


bayesian network

* Gaussian processes
  * way of  representing/approximating unknown functions 
  * $\text{Definition: } p(f) \text{ is a Gaussian Process if for any finite subset } \\\{x_1, ..., x_n\} \sub X \text{, the marginal distribution over that subset } p(f) \\ \text{ is multivariate Gaussian}$ 
* bayesian neural network
  * bayesian?
    * neural network: parameters are fixed
    * bayesian neural network: parameters are given in some probabilistic distribution 
    * dealing with parameter uncertainty, possibly structure uncertainty 
  * bayes rule:
    * $P(hypothesis | data) = \frac{P(hypothesis) P(data | hypothesis)}{\sum _h P(h) P(data | h)} $ 
    * P(hypothesis)
      * we should have some hypothesis about the data through a probability distribution 
      * this is the prior 
    * P(data|hypothesis)
      * this is the likelihood 
    * sum of pausible hypothesis
    * this is learning from data 
    * hypothesis: anything uncertain
    * data: measured things 
  * Data: $D = \{(x^{(n)}, y^{(n)})\} ^N _{n=1} = (X, y)$ 
  * Parameters (weghts): $\theta$ 
  * hyper-parameter: $\alpha$ 
    * precision of the gaussian 
  * prior: $p(\theta | \alpha)$ 
    * way of representing some range 
  * posterior: $p(\theta | \alpha, D) \propto p(y|X, \theta)p(\theta | \alpha)$  
    * observe the data and get posterior from some algorithm 
    * bayesian neural network is all about estimating the posterior
  * evidence 
  * prediction: $p(y\prime | D, x\prime, \alpha) = \int p(y\prime | x\prime, \theta)p(\theta | D, \alpha) d\theta$ 
    * average over the parameter values to make your prediction 
* ARD (Automatic Relevance Determination) using Bayesian Network
  * $p(w_{dj} | \alpha _d) = N(0, \alpha_d ^{-1})$ where weighs from feature $x_d$ have variance $\alpha _d ^{-1}$ 
  * $a_d \rightarrow \infin \text{ then variance }\rightarrow 0 \text{ then weights }\rightarrow 0$ 
    * then the feature $d$ is irrelevant
  *  $a_d \text{ is finite number then finite variance then weight can vary}$ 
    * then the feature $d$ is relevant
* 



* probabilities in machine learning 
  * data distribution (or evidence): $P(x)$ 
  * likelihood: $P(x|c)$ 
  * posterior: $P(c|x)$ 
  * prior: $P(c)$ 



* posterior approximation algorithms 
  * logistic regression + Laplace approximation can be an approach  
    * logistic regression:
      * likelihood: probablility of y given x
        * $p(y|x, w) = \Pi _{i=1} ^{N} \sigma (x_i^Tw)^{y_i}(1-\sigma (x_i^Tw))^{1-y_i}$  
      * MLE (maximum likelihood estimation): finding the parameter that maximizes the likelihood 
        * $w^* = \arg \max _* p(y|x,w)$ 
    * bayesian logistic regression:
      * need to select prior distribution of $w$ 
        * $p(w) = N(w; \mu , \sum)$ 
      * then use the prior distribution to derive the posterior distribution $p(w|D)$ 
        * $p(w|D) = \frac{p(y|x, w) p(w)}{p(y|x)}$ 
        * but here we don't know what $p(y|x)$ is 
    * using laplace approximation to approximate the posterior distribution $p(\theta | D)$ (training)
    * 1. we have a prior distribution
      2. define logarithmic unnormalized posteriori distribution
      3. find $\hat \theta $ that maximizes the distribution above
         1. how? just use some optimization techniques 
      4. find hessian of the distribution above 
      5. $p(\theta| D) \sim  N(\hat \theta, H ^{-1})$ 
    * prediction:

* Mean-Field Approximation 
  * Variational Bayesian methods: familily of methods for approximating intractable integrals arising in Bayesian inference or machine learning problems 
    * Mean-Field Approximation is one of simplest VB methods
  * find the Normal distribution that is closest to the actual distribution
    * use the Reverse KL divergence as the distance metric between distributions
      * Reverse KL divergence: 
        * measures the amount of information required to distort the approximated distribution to original distribution
        * $K L\left(Q_{\phi}(Z | X) \| P(Z | X)\right)=\sum_{z \in Z} q_{\phi}(z | x) \log \frac{q_{\phi}(z | x)}{p(z | x)}$ 
          * $Q_\phi (Z|X)$ is the approximated distribution
          * $P(Z|X)$ is the original distribution 
          * $\phi$ is the parameter we want to change to optimize the equation 
        * KL divergence 
          * Kullback-Leibler divergence is the measure of the difference between two probability distributions
          * KL divergence from $Q$ to $P$ is written like:
            * $D_{KL}(P||Q)= E_{x~P}[\log \frac{P(X)}{Q(X)}]$  
            * KL is not symmetric (forward, reverse KL)
              * $D_{KL}(P||Q)\ne D_{KL}(Q||P)$ 
              * because the resulting derivation needs different functions 
              * forward KL v.s. reverse KL 
                * forward KL: makes the distribution wide so that there is minimal area that $q(x) = 0$ and $p(x) > 0$  
                * reverse KL: makes the distribution shallower so that the $p(x)$ includes all $q(x)$ 





* most bayesian methods can be categorized into two methods:

  * sampling

    * major paradigm: MCMC (Markov Chain Monte Carlo) sampling 

      * basic idea:

        * paradigm of algorithms based on the idea of Markov Chain and Monte Carlo properties 
        * Monte Carlo: sampling using statistical method
        * Markov Chain: the next sample depends only on the current sample. Markov property
          * Markov property: next state only depends on current state and not the state before.
          * how you pick the next sampling point based on current sampling point leads to different algorithms 

      * goal: sample from p, or approximate some expectation E[f(x)] where (x ~ p)

      * problem background:

        * when probability distribution $p$ is very high dimensional and complicated, analytical sampling is impossible. 
        * when solving very complex integrals, computers use sampling without actually doing the integral 

      * Metropolis-Hasting algorithm

        * the most common/basic algorithm. most MCMC algorithms are variations of this algorithm. 

        * given that the target distribution is $p(x)$, if you can calculate some distribution $f(x)$ that is proportional to $p(x)$, then you can apply MCMC sampling algorithm to distribution $p(x)$ 

          * e.g. can sample the posterior function by only using the likelihood and prior term in the bayesian rule 

        * you need a conditional distribution $q(x^t|x)$ called "proposal distribution" to sample the next state $x$ you are trying to sample 

        * if you are trying to sample a new sample from 

        * pseudo-code:
      
          * ```
            initialize x[0]
            for i = 0 to N - 1:
            	sample u ~ U[0, 1]
            	sample x_new ~ q(x_new|x[i])
            	if u < A(x[i], x_new) = min(1, p(x_new)q(x[i]|x_new) / (p(x[i])q(x_new|x[i]))):
            		x[i+1] = x_new
            	else:
        		x[i+1] = x[i]
            ```

          * 

      * gibbs sampling algorithm 
    
        * sampling using conditional probability 

  * variational inference 
  
    * variational inference is the method used to approximate the posterior probability distribution by minimizing KL divergence
    * 



* MLE (maximum likelihood estimation)
  * MLE is a machine learning method that estimates some parameter that maximizes the likelihood 
  * compare $p(z|a)$ and $p(z|b)$ to choose which class $z$ belongs to

* MAP (Maximum A Posteriori) approximation
  * MAP is also a machine learning method that estimates some parameter that maximizes the posterior probability
  * compare $p(a|z)$ and $p(b|z)$ to choose which class $z$ belongs to 





* we want to estimate the confidence of some neural network's output.
* we want to use bayesian neural network for this task.
  * basic idea of bayesian neural network is to replace the parameters of neural network with probability distributions. 
  * this way, we can estimate the output distribution of some neural network. 
    * if the output distribution is wide, then the neural network's confidence is not so good.
    * if the output distribution is narrow, then the neural network's output is very confident.
  * so, how do we the probability distribution of the weights?
  * here we use the bayes theorem!
  * $p(w|D) = P(D|w)P(w)/P(D)$ 
    * here, $P(D|w)$ is the likelihood
      * likelihood is the probability distribution of the data 
    * and $P(w)$ is the prior.
      * prior is the distribution where the parameters are likely to be in. 
* here, we may have these questions:
  * firstly and most importantly, how do we get the posterior?
    * one example is given by professor: logistric regression + laplace approximation.
  * secondly, what is a good prior with/without prior knowledge?
  * how can we use bayesian neural network to estimate the confidence of a neural network's output?
* 
