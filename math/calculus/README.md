# Calculus Project Overview

## General Topics
- **Summation and Product notation**: Understand and apply mathematical notations for summation and products.
- **Series**: What are they and what are the common types?
- **Derivatives**: Understanding basic derivative concepts.
- **Product Rule**: Learn how this rule applies in differentiation.
- **Chain Rule**: Another fundamental rule in calculus for deriving functions.
- **Common Derivative Rules**: Overview of frequently used derivative rules.
- **Partial Derivatives**: Introduction to differentiation with respect to one variable while keeping others constant.
- **Indefinite Integrals**: Basic concepts and applications.
- **Definite Integrals**: Learn to calculate the area under curves.
- **Double Integrals**: Extend integration concepts to two-dimensional problems.

## Tasks

### 0. Sigma is for Sum
- **Objective**: Calculate the sum \(\sum_{i=2}^{5} i\)
- **Possible Answers**:
  - 3 + 4 + 5
  - 3 + 4
  - 2 + 3 + 4 + 5
  - 2 + 3 + 4
- **File**: `0-sigma_is_for_sum`

### 1. The Greeks pronounce it sEEgma
- **Objective**: Compute \(\sum_{k=1}^{4} 9i - 2k\)
- **Possible Answers**:
  - 90 - 20
  - 36i - 20
  - 90 - 8k
  - 36i - 8k
- **File**: `1-seegma`

### 2. Pi is for Product
- **Objective**: Find the product \(\prod_{i = 1}^{m} i\)
- **Possible Answers**:
  - (m - 1)!
  - 0
  - (m + 1)!
  - m!
- **File**: `2-pi_is_for_product`

### 3. The Greeks pronounce it pEE
- **Objective**: Evaluate \(\prod_{i = 0}^{10} i\)
- **Possible Answers**:
  - 10!
  - 9!
  - 100
  - 0
- **File**: `3-pee`

### 4. Hello, derivatives!
- **Objective**: Derive \(\frac{dy}{dx}\) where \(y = x^4 + 3x^3 - 5x + 1\)
- **Possible Answers**:
  - 3x^3 + 6x^2 - 4
  - 4x^3 + 6x^2 - 5
  - 4x^3 + 9x^2 - 5
  - 4x^3 + 9x^2 - 4
- **File**: `4-hello_derivatives`

### 5. A log on the fire
- **Objective**: Differentiate \(\frac{d (x \ln(x))}{dx}\)
- **Possible Answers**:
  - \(\ln(x)\)
  - \(\frac{1}{x} + 1\)
  - \(\ln(x) + 1\)
  - \(\frac{1}{x}\)
- **File**: `5-log_on_fire`

### 6. It is difficult to free fools from the chains they revere
- **Objective**: Find the derivative \(\frac{d (\ln(x^2))}{dx}\)
- **Possible Answers**:
  - \(\frac{2}{x}\)
  - \(\frac{1}{x^2}\)
  - \(\frac{2}{x^2}\)
  - \(\frac{1}{x}\)
- **File**: `6-voltaire`

### 7. Partial truths are often more insidious than total falsehoods
- **Objective**: Compute \(\frac{\partial f(x, y)}{\partial y}\) where \(f(x, y) = e^{xy}\)
- **Possible Answers**:
  - \(e^{xy}\)
  - \(ye^{xy}\)
  - \(xe^{xy}\)
  - \(e^{x}\)
- **File**: `7-partial_truths`

### 8. Put it all together and what do you get?
- **Objective**: Determine \(\frac{\partial^2}{\partial y \partial x}(e^{x^2y})\)
- **Possible Answers**:
  - 2x(1+y)e^{x^2y}
  - 2xe^{2x}
  - 2x(1+x^2y)e^{x^2y}
  - e^{2x}
- **File**: `8-all-together`

### Additional Tasks (9-17)
Further tasks include creating functions to compute squared summations, derivatives of polynomials, and integrals. Each task builds on the understanding of basic calculus principles applied through Python programming.

## Repository Information
- **GitHub Repository**: holbertonschool-machine_learning
- **Directory**: math/calculus

This README provides an overview of the tasks and learning objectives associated with this project. For each task, students are required to write code, make calculations, or both, adhering to the given specifications and requirements.
