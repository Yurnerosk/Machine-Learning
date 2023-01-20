# Machine-Learning
Few machine learning algorithm examples

In here you'll find various machine learning exercises found around the internet. I do not own them.

## Gradient Descent
In this example, we have a training set containing sales and spendings with media.

The model is then trained with a training set containing 200 observations, each labeled as $(Spendings_i , Sales_i)$. The objective of this training is to create a linear model (**linear regression**), that looks like something like this:

$$ f(x) = wx + b $$

notice that the optimal values for **w** and **b** are yet to be found, and to do that we must look for two values that minimize a **loss function**, the mean squared error **l**:

$$ l = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \left( wx_i + b \right) \right)^2 $$

Gradient Descent starts with calculating the partial derivative for every parameter:

$$\frac{\partial l}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} 2\left( y_i - \left( wx_i + b \right) \right)*\left( -xi \right)$$

$$\frac{\partial l}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} 2\left( y_i - \left( wx_i + b \right) \right)*\left( -1 \right)$$

<p align="right">Remember the chain rule? When $h(x) = f(g(x))$, then $h'(x) =f'(g(x))\cdot g'(x)$</p> 

Each time Gradient Descent use the training set to update each parameter is called an **epoch**. At the first one, we initialize $w\gets 0$ and $b\gets 0$, and at each epoch, the update is given by:

$$ w\gets w - \alpha\frac{\partial l}{\partial x} $$ 

$$ b\gets b - \alpha\frac{\partial l}{\partial b} $$ 

, where $\alpha$ is the **learning rate** (it controls the size of an update).

It is important to understand that, in order to **minimize the objective function**:
- **positive** derivative means the square error **rises**     in such point. Thus the substraction, to move the variable to the left (opposite) direction.
- **negative** derivative means the square error **decreases** in such point. Thus the substraction, to move the variable to the right (opposite) direction.

With the update done, a new epoch is set with the new updated values for **w** and **b**. It's noted that after some epochs those values do not change that much, and that's the time to **stop**.
