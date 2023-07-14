# **Experiments**

The folder consists of the experiments that we performed for the validity of the RSQTOA framework. These experiments 
are specifically concerned with the adaptability of RSQTOA framework to produce approximate model using dataset. In 
order to measure the performance and effectiveness of RSQTOA framework with different approaches such `Grid Sampling` 
and `interpolation` we perform two separate experiments. Consequently, we compare the performance of the RSQTOA based
approaches with the most prominently used modelling techniques such as Artificial Neural Networks, ElasticNet and Linear 
Regression.

Below are listed the two experiment performed during the course of the study:

### Experiments
- **[Synthetic Data]**: This experiment was performed as a baseline to gauge the performance of RSQTOA framework. As it 
aims to separate dimensions. It makes sense to see if it is able to do se. Hence, we perform the experiment where in we 
create a synthetic dataset using an equation. Using this methodology gives us a good idea about the dimension approximation 
capability of the framework. [Synthetic Data] file consist of all the necessary background information needed to 
understand the steps taken during the experiment. Furthermore, it gives a brief overview of the directory structure as well
as how to replicate and use the RSQTOA framework with dataset. 
- **[Yelp Data]**: This experiment was performed as a follow-up experiment. In this experiment we aim to gauge the 
performance of RSQTOQ framework with a real world dataset and real world usecase. For this experiment we intended to 
use [Yelp Academic Dataset] for predicting review helpfulness. For more detailed background and rationale behind the 
experiment please refer to the accompanying [report]. [Yelp Data] file consist of all the necessary background 
information needed to understand the steps taken during the experiment. Furthermore, it gives a brief overview of the
directory structure as well as how to replicate and use the RSQTOA framework with dataset.

### Utils
- **[Commons]**: contains utils and wrapper methods and classes that we used to perform the experiments.

[Synthetic Data]: synthetic/README.md
[Yelp Data]: yelp/README.md
[Commons]: commons/
[Yelp Academic Dataset]: https://www.yelp.com/dataset
[report]: ./../report/02_RSQTOA_Framework_Dataset_Adoption_Ashish_Rajani.pdf
