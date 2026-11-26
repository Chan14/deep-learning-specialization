# deep-learning-specialization

My work through the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) (DeepLearning.AI, taught by Andrew Ng): the programming assignments, my own implementations and experiments, and the reusable code that came out of them.

**Jump to:** [Course 1](#course-1--neural-networks-and-deep-learning) · [Course 2](#course-2--improving-deep-neural-networks-hyperparameter-tuning-regularization-and-optimization) · [Beyond the courses](#beyond-the-courses) · [Working in this repo](#working-in-this-repo)

In each week, the course's own material comes first. **My code** (my own implementations) and **My experiments** (small notebooks and scripts that build intuition for one idea) follow it.

## Course 1 : [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)

- [Week 1 : Introduction to deep learning](course1_neural_networks_and_deep_learning/w1_introduction_to_deep_learning)
  - [Practice quiz: Introduction to deep learning](course1_neural_networks_and_deep_learning/w1_introduction_to_deep_learning/practice_quizzes/p1_introduction_to_deep_learning)

- [Week 2 : Neural networks basics](course1_neural_networks_and_deep_learning/w2_neural_networks_basics)
  - [Practice quiz: Neural network basics](course1_neural_networks_and_deep_learning/w2_neural_networks_basics/practice_quizzes/p1_neural_network_basics)
  - [Programming Assignments](course1_neural_networks_and_deep_learning/w2_neural_networks_basics/programming_assignments)
    - [Python Basics with NumPy (optional)](course1_neural_networks_and_deep_learning/w2_neural_networks_basics/programming_assignments/a1_python_basics_with_numpy/Python_Basics_with_Numpy.ipynb)
    - [Logistic Regression with a Neural Network Mindset](course1_neural_networks_and_deep_learning/w2_neural_networks_basics/programming_assignments/a2_logistic_regression_as_a_neural_network/Logistic_Regression_with_a_Neural_Network_mindset.ipynb)
      - My version: [examples as rows](course1_neural_networks_and_deep_learning/w2_neural_networks_basics/programming_assignments/a2_logistic_regression_as_a_neural_network/logistic_regression_examples_as_rows.ipynb): each example stored as a row, a numerically stable sigmoid, and the cross-entropy cost computed from the logits so it never takes $\log 0$

- [Week 3 : Shallow neural networks](course1_neural_networks_and_deep_learning/w3_shallow_neural_networks)
  - [Practice quiz: Shallow neural networks](course1_neural_networks_and_deep_learning/w3_shallow_neural_networks/practice_quizzes/p1_shallow_neural_networks)
  - [Programming Assignment](course1_neural_networks_and_deep_learning/w3_shallow_neural_networks/programming_assignments)
    - [Planar Data Classification with One Hidden Layer](course1_neural_networks_and_deep_learning/w3_shallow_neural_networks/programming_assignments/a1_planar_data_classification/Planar_data_classification_with_one_hidden_layer.ipynb)
  - [My experiments](course1_neural_networks_and_deep_learning/w3_shallow_neural_networks/experiments)
    - [ReLU vs sigmoid gradient on a shared bias](course1_neural_networks_and_deep_learning/w3_shallow_neural_networks/experiments/relu_vs_sigmoid_bias_gradient.ipynb): interactive
    - [ReLU vs sigmoid gradient on an early weight](course1_neural_networks_and_deep_learning/w3_shallow_neural_networks/experiments/relu_vs_sigmoid_weight_gradient.ipynb): interactive, path by path
    - [A 2D grid through one linear layer vs linear → ReLU → linear](course1_neural_networks_and_deep_learning/w3_shallow_neural_networks/experiments/grid_through_relu_layer.py)
    - [A 2D grid through two layers, step by step](course1_neural_networks_and_deep_learning/w3_shallow_neural_networks/experiments/grid_through_two_layers.py)
    - [A 2D grid through two layers, animated](course1_neural_networks_and_deep_learning/w3_shallow_neural_networks/experiments/grid_through_two_layers_animated.ipynb)

- [Week 4 : Deep neural networks](course1_neural_networks_and_deep_learning/w4_deep_neural_networks)
  - [Practice quiz: Key concepts on deep neural networks](course1_neural_networks_and_deep_learning/w4_deep_neural_networks/practice_quizzes/p1_key_concepts_on_deep_neural_networks)
  - [Programming Assignments](course1_neural_networks_and_deep_learning/w4_deep_neural_networks/programming_assignments)
    - [Building your Deep Neural Network: Step by Step](course1_neural_networks_and_deep_learning/w4_deep_neural_networks/programming_assignments/a1_deep_neural_network_step_by_step/Building_your_Deep_Neural_Network_Step_by_Step.ipynb)
    - [Deep Neural Network: Application](course1_neural_networks_and_deep_learning/w4_deep_neural_networks/programming_assignments/a2_deep_neural_network_application/Deep_Neural_Network_Application.ipynb)
  - [My code](course1_neural_networks_and_deep_learning/w4_deep_neural_networks/my_code)
    - [L-layer network in NumPy](course1_neural_networks_and_deep_learning/w4_deep_neural_networks/my_code/l_layer_network_numpy.ipynb): binary cross-entropy computed from the logits, backprop that starts from $dZ^{[L]} = A^{[L]} - Y$, and early stopping on the cost

**[Certificate of Completion](course1_neural_networks_and_deep_learning/certificate/course1_certificate.pdf)**

## Course 2 : [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network)

- [Week 1 : Practical aspects of deep learning](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning)
  - [Lecture notes](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/lecture_notes/c2_w1_lecture_notes.pdf)
  - [Practice quiz: Practical aspects of deep learning](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/practice_quizzes/p1_practical_aspects_of_deep_learning)
  - [Programming Assignments](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/programming_assignments)
    - [Initialization](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/programming_assignments/a1_initialization/Initialization.ipynb)
    - [Regularization](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/programming_assignments/a2_regularization/Regularization.ipynb)
    - [Gradient Checking](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/programming_assignments/a3_gradient_checking/Gradient_Checking.ipynb)
  - [My experiments](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/experiments)
    - [Dropout on data built to overfit](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/experiments/dropout_intuition/dropout_experiment.py): trains with and without dropout and plots both
    - [Dropout, step by step](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/experiments/dropout_intuition/dropout_stepwise.ipynb)
    - [Normalizing 1D and 2D inputs](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/experiments/normalization_intuition/input_normalization_plots.ipynb)
    - [Gradient descent on unscaled vs normalized features](course2_improving_deep_neural_networks/w1_practical_aspects_of_deep_learning/experiments/normalization_intuition/gradient_descent_unscaled_vs_normalized.ipynb)

- [Week 2 : Optimization algorithms](course2_improving_deep_neural_networks/w2_optimization_algorithms)
  - [Lecture notes](course2_improving_deep_neural_networks/w2_optimization_algorithms/lecture_notes/c2_w2_lecture_notes.pdf)
  - [Practice quiz: Optimization algorithms](course2_improving_deep_neural_networks/w2_optimization_algorithms/practice_quizzes/p1_optimization_algorithms)
  - [Programming Assignment](course2_improving_deep_neural_networks/w2_optimization_algorithms/programming_assignments)
    - [Optimization Methods](course2_improving_deep_neural_networks/w2_optimization_algorithms/programming_assignments/a1_optimization_methods/Optimization_methods.ipynb)
  - [My code](course2_improving_deep_neural_networks/w2_optimization_algorithms/my_code)
    - [Momentum as an exponentially weighted average of gradients](course2_improving_deep_neural_networks/w2_optimization_algorithms/my_code/momentum_as_ewa_of_gradients.py): batch GD, SGD, and SGD with momentum in a narrow ravine, with each velocity update split into the parts that cancel and the parts that add up
    - [Batch GD vs SGD vs mini-batch on California Housing](src/dlspec/gd_variants): the package `dlspec.gd_variants`, described under [Beyond the courses](#reusable-package-dlspec)

- [Week 3 : Hyperparameter tuning, batch normalization and programming frameworks](course2_improving_deep_neural_networks/w3_hyperparameter_tuning_batch_norm_and_frameworks)
  - [Lecture notes](course2_improving_deep_neural_networks/w3_hyperparameter_tuning_batch_norm_and_frameworks/lecture_notes/c2_w3_lecture_notes.pdf)
  - [Practice quiz: Hyperparameter tuning, batch normalization, programming frameworks](course2_improving_deep_neural_networks/w3_hyperparameter_tuning_batch_norm_and_frameworks/practice_quizzes/p1_hyperparameter_tuning_batch_norm_and_frameworks)
  - [Programming Assignment](course2_improving_deep_neural_networks/w3_hyperparameter_tuning_batch_norm_and_frameworks/programming_assignments)
    - [TensorFlow Introduction](course2_improving_deep_neural_networks/w3_hyperparameter_tuning_batch_norm_and_frameworks/programming_assignments/a1_tensorflow_introduction/Tensorflow_introduction.ipynb)
  - [My experiments](course2_improving_deep_neural_networks/w3_hyperparameter_tuning_batch_norm_and_frameworks/experiments)
    - [Sampling hyperparameters on a log scale](course2_improving_deep_neural_networks/w3_hyperparameter_tuning_batch_norm_and_frameworks/experiments/hyperparameter_log_scale_sampling.ipynb): learning rate in $[10^{-4}, 1]$, and momentum $\beta$ through $1 - \beta$
    - [Batch norm by hand](course2_improving_deep_neural_networks/w3_hyperparameter_tuning_batch_norm_and_frameworks/experiments/batch_norm_by_hand.ipynb): normalizing $Z$ per unit, and why scaling $W$ doesn't change the normalized result

**[Certificate of Completion](course2_improving_deep_neural_networks/certificate/course2_certificate.pdf)**

## Beyond the courses

### Reusable package: `dlspec`

[`dlspec.gd_variants`](src/dlspec/gd_variants): batch GD, true SGD, and mini-batch SGD, written by hand in NumPy. A 3-layer ReLU MLP regresses California Housing prices. The three methods differ only in the batch iterator handed to the training loop.

| Module                                                | What it does                                                                                          | Tests                                                      |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [data.py](src/dlspec/gd_variants/data.py)             | Loads California Housing, splits it into train/dev/test, and normalizes with training statistics only | [test_data.py](tests/gd_variants/test_data.py)             |
| [model.py](src/dlspec/gd_variants/model.py)           | He/Xavier init, forward, and backward for the MSE data loss                                           | [test_model.py](tests/gd_variants/test_model.py)           |
| [losses.py](src/dlspec/gd_variants/losses.py)         | MSE data loss plus an L2 penalty on weights only, reported separately                                 | [test_losses.py](tests/gd_variants/test_losses.py)         |
| [optimizer.py](src/dlspec/gd_variants/optimizer.py)   | One vanilla GD step; the L2 gradient goes to weights only, and the caller's gradients aren't mutated  | [test_optimizer.py](tests/gd_variants/test_optimizer.py)   |
| [trainer.py](src/dlspec/gd_variants/trainer.py)       | The three batch iterators and a method-agnostic training loop with early stopping                     | [test_trainer.py](tests/gd_variants/test_trainer.py)       |
| [grad_check.py](src/dlspec/gd_variants/grad_check.py) | Centered-difference check of `model.backward`                                                         | [test_grad_check.py](tests/gd_variants/test_grad_check.py) |

### Explorations

Topics the specialization doesn't teach.

- [Linear algebra](explorations/linear_algebra)
  - [SVD basics](explorations/linear_algebra/svd_basics.ipynb)
- [Statistics and probability](explorations/stats_and_probability)
  - [Entropy, cross-entropy, and KL divergence](explorations/stats_and_probability/kl_divergence.ipynb)
- [PyTorch fundamentals](explorations/pytorch_fundamentals)
  - [Tensors and autograd](explorations/pytorch_fundamentals/tensors_and_autograd.ipynb)

## Working in this repo

### Repository layout

```
deep-learning-specialization/
├── pyproject.toml          # package metadata, dependencies, pytest and ruff config
├── requirements.txt        # exact pinned snapshot of dl-spec-env
├── .python-version         # pinned interpreter (3.12)
├── course1_neural_networks_and_deep_learning/
│   ├── certificate/
│   └── w1_.../ w2_.../ w3_.../ w4_.../    # one folder per week
├── course2_improving_deep_neural_networks/
├── src/dlspec/             # my reusable, tested code
├── tests/                  # mirrors src/dlspec/
└── explorations/           # topics the specialization doesn't teach
```

**Inside each week, the same folder names mean the same thing:**

| Folder | What's in it |
| ------ | ------------ |
| `lecture_notes/` | The course's lecture slides for that week |
| `practice_quizzes/pN_<topic>/` | Screenshots of practice quizzes |
| `programming_assignments/aN_<name>/` | A course assignment with its utilities, tests, images, and data, plus my redo and my own versions of it |
| `my_code/` | My own implementations and scripts for that week |
| `experiments/` | My small notebooks and scripts that build intuition for one idea |
| `scratch/` | Rough work on that week's topics, not linked from this README |

**Rules that keep this layout sound**

1. Assignment folders are self-contained. A notebook imports its utilities
   (`public_tests.py`, `testCases.py`, ...) from its own folder and reads
   `images/` and `datasets/` next to it. Move an assignment folder only as a
   whole. Course files keep the course's own names.
2. My own version of an assignment that uses the assignment's data stays in
   the assignment folder, so it can read `datasets/` next to it.
3. Code worth importing from more than one place goes in `src/dlspec/`, with
   tests in the mirrored `tests/` path. Notebooks import it as
   `from dlspec.<module> import ...`, and never use `sys.path` hacks.
4. Scripts find their files with `Path(__file__).resolve().parent`, so they
   work from any working directory.
5. Course datasets (`*.h5`, `*.mat`) and videos stay on my machine. They are
   gitignored, so a fresh clone needs them copied in from Coursera.
6. Any folder named `untracked/`, at any depth, is ignored by git.

### Setup

The environment is the pyenv virtualenv `dl-spec-env` (Python 3.12).

One time, from the repo root:

```bash
pyenv activate dl-spec-env
pip install -e ".[dev]"            # editable install of dlspec + dev tools
nbstripout --install               # strip notebook outputs on commit; local files keep them
pytest                             # expect all tests to pass
```

To rebuild the exact environment on a new machine:
`pip install -r requirements.txt && pip install -e . --no-deps`.
After changing the environment, refresh the snapshot with
`pip freeze --exclude-editable > requirements.txt`.

### Everyday commands

```bash
pyenv activate dl-spec-env                       # once per terminal
pytest                                           # run all tests
pytest tests/gd_variants/test_trainer.py         # run one module's tests
python -m dlspec.gd_variants.grad_check          # print the gradient check table
python course2_improving_deep_neural_networks/w2_optimization_algorithms/my_code/momentum_as_ewa_of_gradients.py
ruff check src tests                             # lint my package
```

### Where new work goes

| Content type | Where it goes |
| ------------ | ------------- |
| A course assignment (downloaded) | `courseN_.../wN_.../programming_assignments/aN_.../` |
| My redo or my own version of that assignment | the same folder |
| Lecture slides | `courseN_.../wN_.../lecture_notes/` |
| My own implementation or script for a week | `courseN_.../wN_.../my_code/` |
| A notebook or script that builds intuition for one idea | `courseN_.../wN_.../experiments/` |
| Rough work on a week's topic | `courseN_.../wN_.../scratch/` |
| A function I'll reuse across notebooks | `src/dlspec/...` + `tests/...` |
| Anything the specialization doesn't teach | `explorations/<topic>/` |
| Anything that must never reach GitHub | any `untracked/` folder |
