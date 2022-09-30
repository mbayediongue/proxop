
# Proximity Operator Repository 

Proximity operators have become increasingly important tools as basic 
building blocks of proximal splitting algorithms, a class of algorithms 
that decompose complex composite convex optimization methods into simple 
steps involving one of the functions present in the model. This package provides
implementations of the proximity operator of various
functions (function of scalar variable or multivariate,convex and non-convex functions, 
indicator functions...). 

For more information see the [full project](http://proximity-operator.net/) containing technical details, a tutorial and code implantation in matlab.

## Installation

To install package (require python 3.8.0 or a later version):

    pip install proxop

To update, one should add the option "--upgrade":

    pip install proxop --upgrade

## How to use it ?

To import the package:

    import proxop

Please visit our [website](http://proximity-operator.net/)  for more tutorial and more details.

## Examples:

Evaluates the function 'AbsValue':
    >>> import proxop
    >>> proxop.AbsValue()(-3)
    3

Use a scale factor 'gamma':
    >>> proxop.AbsValue(gamma=2)(-3)
    6

When the input is a vector, the result is the sum of the outputs obtained by applying the function to each element :

     >>> from proxop import AbsValue
     >>> import numpy as np
     >>> AbsValue()( np.array([-1, 2, 3., -4.]) )
     10.0
  
Compute the proximity operator by using the method 'prox' :

    >>> AbsValue().prox( 3)
    2
    >>> AbsValue().prox(np.array([ -3., 1., 6., 3.]))
    array([-2.,  0.,  5., 2.])


Use a scale factor 'gamma'>0 to commute the proximity operator of  th function "gamma*f" :

    >>> AbsValue(gamma=2).prox([ -3., 1., 6.])
     array([-1.,  0.,  4.])

## Example 2: Projection onto a set
Since proximity operator is a generalization of the notion of projection onto a (convex) set,
one can easily interpret the result with an indicator function. For example, the affine barrier is defined as:

                       / -log(b- a.T*x)    if u.T*x < b
                f(x) =|
                       \   + inf            otherwise


     >>> import numpy as np
     >>> from proxop import AffineBarrier
     >>>
     >>> x=np.array([1,2,3])
     >>> a= np.array([-1, 5, 3])
     >>> b= 3.5
     >>> AffineBarrier(a,b)(x)
     inf  

The result below is infinite, which means 'x' does not belong to the affine set.

Projection of 'x' onto the affine set:

     >>> px= AffineBarrier(x,b).prox(x) 
     >>> px
     >>> AffineBarrier(a,b)(px)
     0.61828190224889

As expected, the result is finite, meaning the projection of 'x' belongs to the affine set.

## Example 3: Matrix variable

     >>> x=np.arange(6)
     >>> x=x.reshape((2,3))
     >>> x
     array([[0, 1, 2],
           [3, 4, 5]])
     >>> a =np.ones_like(x)
     >>> a[0,:]=2
     >>> a
     array([[2., 2., 2.],
            [1., 1., 1.]])
     >>> b=np.array([-1, 2, 4])

Set 'axis=0' to process along the rows of the matrix 'x' (note the dimension of 'b'
must be compatible with the shape of 'x'):

     >>> AffineBarrier(a,b, axis=0)(x)
     inf

 Projection of x onton the affine set:

     >>> px = AffineBarrier(a,b, axis=0).prox(x)  
     >>> AffineBariier(x,b, axis=0)(px)
      0.157704693902156
