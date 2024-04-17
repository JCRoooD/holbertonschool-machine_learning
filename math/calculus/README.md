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

### 9. Our life is the sum total of all the decisions we make every day, and those decisions are determined by our priorities
- **Objective**: Write a function `def summation_i_squared(n):` that calculates \(\sum_{i=1}^{n} i^2\).
- **Conditions**:
  - `n` is the stopping condition.
  - Return the integer value of the sum.
  - Return `None` if `n` is not a valid number.
  - No loops are allowed.
- **File**: `9-sum_total.py`

### 10. Derive happiness in oneself from a good day's work
- **Objective**: Write a function `def poly_derivative(poly):` that calculates the derivative of a polynomial.
- **Details**:
  - `poly` is a list of coefficients representing a polynomial where the index represents the power of x.
  - Example: if \(f(x) = x^3 + 3x + 5\), `poly` is `[5, 3, 0, 1]`.
  - Return `None` if `poly` is not valid.
  - Return `[0]` if the derivative is 0.
  - Return a new list of coefficients representing the derivative of the polynomial.
- **File**: `10-matisse.py`

### 11. Good grooming is integral and impeccable style is a must
- **Objective**: Solve and provide the general formula for the integration of polynomial terms.
- **Possible Answers**:
  - \(3x^2 + C\)
  - \(x^4/4 + C\)
  - \(x^4 + C\)
  - \(x^4/3 + C\)
- **File**: `11-integral`

### 12. We are all an integral part of the web of life
- **Objective**: Provide the integral result for exponential functions involving y.
- **Possible Answers**:
  - \(e^{2y} + C\)
  - \(e^y + C\)
  - \(e^{2y}/2 + C\)
  - \(e^y/2 + C\)
- **File**: `12-integral`

### 13. Create a definite plan for carrying out your desire and begin at once
- **Objective**: Solve a specific calculus problem (task specifics not provided in the details).
- **Possible Answers**:
  - 3
  - 6
  - 9
  - 27
- **File**: `13-definite`

### 14. My talents fall within definite limitations
- **Objective**: Determine the bounds or limits of a given function (task specifics not provided in the details).
- **Possible Answers**:
  - -1
  - 0
  - 1
  - undefined
- **File**: `14-definite`

### 15. Winners are people with definite purpose in life
- **Objective**: Calculate the output of a function given specific inputs (task specifics not provided in the details).
- **Possible Answers**:
  - 5
  - 5x
  - 25
  - 25x
- **File**: `15-definite`

### 16. Double whammy
- **Objective**: Solve a calculus problem involving logarithms.
- **Possible Answers**:
  - \(9\ln(2)\)
  - 9
  - \(27\ln(2)\)
  - 27
- **File**: `16-double`

### 17. Integrate
- **Objective**: Write a function `def poly_integral(poly, C=0):` that calculates the integral of a polynomial.
- **Details**:
  - `poly` is a list of coefficients representing a polynomial where the index represents the power of x.
  - Example: if \(f(x) = x^3 + 3x + 5\), `poly` is `[5, 3, 0, 1]`.
  - `C` is an integer representing the integration constant.
  - If a coefficient is a whole number, it should be represented as an integer.
  - Return `None` if `poly` or `C` are not valid.
  - Return a new list of coefficients representing the integral of the polynomial, with the list being as small as possible.
- **File**: `17-integrate.py`

## Repository Information
- **GitHub Repository**: holbertonschool-machine_learning
- **Directory**: math/calculus

This README provides an overview of the tasks and learning objectives associated with this project. For each task, students are required to write code, make calculations, or both, adhering to the given specifications and requirements.
