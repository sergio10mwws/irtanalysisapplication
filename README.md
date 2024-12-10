# IRT Analysis Application

## Description
A Streamlit-based web application for Item Response Theory (IRT) analysis that supports 1PL, 2PL, and 3PL models. The application allows users to upload response data and analyze it using different IRT models.

**Item Response Theory (IRT)** is a statistical framework used to analyze how individuals respond to items in tests, surveys, or questionnaires. It is widely applied in education, psychology, and other fields to ensure that tests accurately measure specific traits, abilities, or attitudes. IRT models describe the probability of a correct response based on the person's ability and item-specific properties.

## The 1PL, 2PL, and 3PL Models
In IRT, the 1PL, 2PL, and 3PL models represent progressively more complex ways to describe the relationship between a person's ability and their likelihood of answering an item correctly. Here's a detailed explanation:

---

### **1PL Model (One-Parameter Logistic)**
- **Purpose:** 
  Used to evaluate a single latent trait or ability, assuming that all items share the same discrimination power (how well an item differentiates between individuals of different abilities).
- **How it works:** 
  The probability \( P(\theta) \) of a correct response is modeled as a logistic function:

  \[
  P(\theta) = \frac{1}{1 + e^{-1.7(\theta - b)}}
  \]

  - \( \theta \): Person's ability level.
  - \( b \): Difficulty of the item.
  - \( 1.7 \): A scaling factor to approximate the normal ogive model.

  The curve shifts horizontally depending on \( b \), representing the difficulty level: higher \( b \) means harder items.

**Example:** A mental math test where all questions have similar power to discriminate between low- and high-ability participants.

---

### **2PL Model (Two-Parameter Logistic)**
- **Purpose:** 
  Extends the 1PL model by adding an item-specific parameter for discrimination, making it more flexible.
- **How it works:**
  The probability \( P(\theta) \) is modeled as:

  \[
  P(\theta) = \frac{1}{1 + e^{-1.7a(\theta - b)}}
  \]

  - \( a \): Discrimination parameter. Higher \( a \) values indicate better differentiation between individuals with slightly different ability levels.
  - \( b \): Difficulty parameter.
  - \( \theta \): Ability level.

  The slope of the curve at \( b \) is determined by \( a \), with steeper slopes for items that are better at discriminating ability differences.

**Example:** An advanced mathematics test where some questions are much better than others at identifying top-performing students.

---

### **3PL Model (Three-Parameter Logistic)**
- **Purpose:** 
  Adds a "guessing" parameter to the 2PL model, acknowledging that individuals might guess the correct answer on multiple-choice items.
- **How it works:**
  The probability \( P(\theta) \) becomes:

  \[
  P(\theta) = c + (1 - c) \frac{1}{1 + e^{-1.7a(\theta - b)}}
  \]

  - \( c \): Guessing parameter, representing the probability of answering correctly by chance (e.g., \( c = 0.25 \) for a 4-option multiple-choice question).

  The lower asymptote of the curve is determined by \( c \), reflecting the chance level of correct responses due to guessing.

**Example:** A multiple-choice test where someone unfamiliar with the material might guess correctly 25% of the time on average.

---

## Why is IRT Useful?
1. **Personalization:** Enables the design of adaptive tests where questions adjust dynamically to the test-taker's ability (e.g., online assessments).
2. **Fairness:** Helps identify and remove biased or poorly functioning items, ensuring tests are valid across different groups.
3. **Precision:** Accurately estimates a person's ability, considering that not all items are equally difficult or effective at measuring differences.

---

## Visualizing the Models
Below is a conceptual representation of how the models differ:
- **1PL:** All curves have the same slope but shift horizontally based on \( b \).
- **2PL:** Curves have varying slopes (discrimination levels).
- **3PL:** Curves have varying slopes and include a lower asymptote due to guessing.

In summary, the 1PL, 2PL, and 3PL models provide increasingly nuanced ways to model response probabilities, enhancing the accuracy and fairness of assessments.

---
## Features

- Support for three IRT models:
  - 1PL (Rasch) Model
  - 2PL Model
  - 3PL Model
- Interactive data upload
- Parameter estimation
- Visual representation of Item Characteristic Curves (ICC)
- Downloadable results
- Ability score calculation (for 1PL model)

---

## Requirements
```bash
pip install streamlit pandas numpy matplotlib torch pyro-ppl py-irt
```

---

## Run
```bash
streamlit run app.py 
```
