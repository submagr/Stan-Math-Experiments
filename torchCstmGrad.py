import torch
import numpy as np


class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return x**2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        jacobian_y_x = torch.DoubleTensor(input.shape[0], input.shape[0])
        for i in range(input.shape[0]):
            jacobian_y_x[i][i] = 2*input[i]

        return torch.matmul(grad_output, jacobian_y_x)


xVal = np.array([2,3,4,5])
x = torch.tensor(xVal, dtype=torch.float64, requires_grad=True)

myFunc = MyFunc.apply
y = myFunc(x)
# y = x**2
print("y = ", y)
print("y expected: ", xVal**2)

y.backward(torch.DoubleTensor([1,1,1,1]))
print("x grad: ", x.grad)
print("x grad expected: ", 2*xVal)
