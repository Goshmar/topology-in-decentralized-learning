## [Beyond spectral gap: The role of the topology in decentralized learning](https://arxiv.org/pdf/2206.03093.pdf)

### Abstract
This repository contains the code and documentation for the paper titled "Beyond Spectral Gap: The Role of Topology in Decentralized Learning." In this paper, we explore the impact of topology in decentralized learning settings, where workers collaborate to optimize machine learning models.

### Description
In data-parallel optimization of machine learning models, workers collaborate to improve their model estimates. More accurate gradients enable them to use larger learning rates and optimize faster. This project focuses on scenarios where all workers sample from the same dataset and communicate over a sparse graph (decentralized). It is essential to note that existing theoretical frameworks fail to fully capture real-world behavior in this context.

Spectral Gap and Empirical Performance: Our research reveals that the 'spectral gap' of the communication graph does not reliably predict its empirical performance in (deep) learning.

Collaboration and Learning Rates: Current theory falls short in explaining how collaboration enables larger learning rates compared to training alone. In fact, it prescribes smaller learning rates, which decrease further as graphs grow in size, failing to account for convergence in infinite graphs.

To address these shortcomings, this project aims to provide a comprehensive understanding of sparsely-connected distributed optimization when workers share the same data distribution. We quantify how the graph topology influences convergence, providing insights into a quadratic toy problem and offering theoretical results for general smooth and (strongly) convex objectives. Our theory aligns with empirical observations in deep learning and accurately describes the relative merits of different graph topologies.

### Contents
 - **code/**: Contains the code of the team's developments within the framework of the use of tools not considered in the article, as an addition to additional research.

 - **deep-learning-experiments/**: Contains the source code for the experiments and simulations discussed in the paper.

 - **docs/**: Contains supplementary documentation, including research papers and reports related to the project.

 - **results**/: Stores the results of experiments and simulations conducted during the research.

### Usage
Please refer to the documentation in the code/directory for detailed information about conducting experiments and reproducing the results discussed in the article and beyond. Also note the file "requirements.txt" for correct reproduction of the results.

### License
This project is open-source and is distributed under the MIT license. Please refer to the LICENSE.txt file for more details on licensing terms and conditions.
