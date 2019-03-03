[HMKCODE](http://hmkcode.com/)

# Backpropagation Step by Step

15 FEB 2018

![](https://hmkcode.github.io/images/ai/backpropagation.png)

If you are building your own neural network, you will definitely need to understand how to train it. Backpropagation is a commonly used technique for training neural network. There are many resources explaining the technique, but this post will explain backpropagation with concrete example in a very detailed colorful steps.	

## Overview

In this post, we will build a neural network with three layers:

- **Input** layer with two inputs neurons
- One **hidden** layer with two neurons
- **Output** layer with a single neuron

![android-tabs](http://hmkcode.github.io/images/ai/nn1.png)

## Weights, weights, weights

Neural network training is about finding weights that minimize prediction error. We usually start our training with a set of randomly generated weights.Then, backpropagation is used to update the weights in an attempt to correctly map arbitrary inputs to outputs.

Our initial weights will be as following: `w1 = `, `w2 = 0.21`, `w3 = 0.12`, `w4 = 0.08`, `w5 = 0.14` and `w6 = 0.15`

![bp_weights](http://hmkcode.github.io/images/ai/bp_weights.png)

## Dataset

Our dataset has one sample with two inputs and one output.

![dataset](http://hmkcode.github.io/images/ai/bp_dataset.png)

Our single sample is as following `inputs=[2, 3]` and `output=[1]`.

![training_sample](http://hmkcode.github.io/images/ai/bp_sample.png)

## Forward Pass

We will use given weights and inputs to predict the output. Inputs are multiplied by weights; the results are then passed forward to next layer.

*Forward pass image*

Matrix Multiplication Details:
$$
\begin{bmatrix}
2&3\\
\end{bmatrix}.\begin{bmatrix}
.31&.16\\
.22&.18\\
\end{bmatrix} = \begin{bmatrix}
1.28&.86\\
\end{bmatrix}.\begin{bmatrix}
.24\\
.17\\
\end{bmatrix} = \begin{bmatrix}
.198\\
\end{bmatrix}
$$
​                                                 2x.31 + 3x.22 = 1.28           1.28*0.24 + 0.86*0.17 = 0.198

​			                         2x.16 + 3x.18 =  .86

## Calculating Error

Now, it’s time to find out how our network performed by calculating the difference between the actual output and predicted one. It’s clear that our network output, or **prediction**, is not even close to **actual output**. We can calculate the difference or the error as following.

![bp_error](http://hmkcode.github.io/images/ai/bp_error.png)

## Reducing Error

Our main goal of the training is to reduce the **error** or the difference between **prediction** and **actual output**. Since **actual output** is constant, “not changing”, the only way to reduce the error is to change **prediction** value. The question now is, how to change **prediction** value?

By decomposing **prediction** into its basic elements we can find that **weights** are the variable elements affecting **prediction** value. In other words, in order to change **prediction** value, we need to change **weights** values.

![bp_prediction_elements](http://hmkcode.github.io/images/ai/bp_prediction_elements.png)

> The question now is **how to change\update the weights value so that the error is reduced?**
> The answer is **Backpropagation!**

## **Backpropagation**

**Backpropagation**, short for “backward propagation of errors”, is a mechanism used to update the **weights** using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). It calculates the gradient of the error function with respect to the neural network’s weights. The calculation proceeds backwards through the network.

> **Gradient descent** is an iterative optimization algorithm for finding the minimum of a function; in our case we want to minimize th error function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point.

![bp_update_formula](http://hmkcode.github.io/images/ai/bp_update_formula.png)

For example, to update `w6`, we take the current `w6` and subtract the partial derivative of **error** function with respect to `w6`. Optionally, we multiply the derivative of the **error** function by a selected number to make sure that the new updated **weight** is minimizing the error function; this number is called **learning rate**.

![update w6](http://hmkcode.github.io/images/ai/bp_w6_update.png)

The derivation of the error function is evaluated by applying the chain rule as following

![finding partial derivative with respect to w6](http://hmkcode.github.io/images/ai/bp_error_function_partial_derivative_w6.png)

So to update `w6` we can apply the following formula

![bp_w6_update_closed_form.png](http://hmkcode.github.io/images/ai/bp_w6_update_closed_form.png)

Similarly, we can derive the update formula for `w5` and any other weights existing between the output and the hidden layer.

![bp_w5_update_closed_form.png](http://hmkcode.github.io/images/ai/bp_w5_update_closed_form.png)

However, when moving backward to update `w1`, `w2`, `w3` and `w4` existing between input and hidden layer, the partial derivative for the error function with respect to `w1`, for example, will be as following.

![finding partial derivative with respect to w1](http://hmkcode.github.io/images/ai/bp_error_function_partial_derivative_w1.png)

We can find the update formula for the remaining weights `w2`, `w3` and `w4` in the same way.

In summary, the update formulas for all weights will be as following:

![bp_update_all_weights](http://hmkcode.github.io/images/ai/bp_update_all_weights.png)

We can rewrite the update formulas in matrices as following

![bp_update_all_weights_matrix](http://hmkcode.github.io/images/ai/bp_update_all_weights_matrix.png)

## Backward Pass

Using derived formulas we can find the new **weights**.

> **Learning rate:** is a hyperparameter which means that we need to manually guess its value.
>
> Delta = prediction - actual = 0.198 - 1 = -0.802
>
> a = Learning Rate = 0.05 (we smartly guess this number)

$$
\begin{bmatrix}w5\\w6\\\end{bmatrix}=\begin{bmatrix}.24\\
.17\\\end{bmatrix} - 0.05(-0.802)\begin{bmatrix}1.28\\.86\\\end{bmatrix} = \begin{bmatrix}.24\\
.17\\\end{bmatrix} - \begin{bmatrix}-.05\\
-.03\\\end{bmatrix} = \begin{bmatrix}.29\\.20\\\end{bmatrix}\\

\begin{bmatrix}w1&w3\\w2&w4\\\end{bmatrix} = \begin{bmatrix}.31&.16\\.22&.18\\\end{bmatrix} - 0.05(-0.802)\begin{bmatrix}2\\3\\\end{bmatrix}\begin{bmatrix}.24&.17\\\end{bmatrix} = \begin{bmatrix}.31&.16\\.22&.18\\\end{bmatrix} -\begin{bmatrix}-.02&-.01\\-.03&-.02\\\end{bmatrix} = \begin{bmatrix}.33&.17\\.25&.20\\\end{bmatrix}
$$



Now, using the new **weights** we will repeat the forward passed

$$
\begin{bmatrix}
2&3\\
\end{bmatrix}.\begin{bmatrix}
.33&.17\\
.25&.20\\
\end{bmatrix} = \begin{bmatrix}
1.41&.94\\
\end{bmatrix}.\begin{bmatrix}
.29\\
.20\\
\end{bmatrix} = \begin{bmatrix}
.27\\
\end{bmatrix}
$$
​                                                2x.33+ 3x.25 = 1.41         1.41x.29+.94x0.20 = 0.27

​			                        2x.17 + 3x.20=  .94

We can notice that the **prediction** `0.27` is a little bit closer to **actual output** than the previously predicted one `0.198`. We can repeat the same process of backward and forward pass until **error** is close or equal to zero.


