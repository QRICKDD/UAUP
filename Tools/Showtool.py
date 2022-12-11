import cv2
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt


def img_show3(img):
    plt.matshow(img)
    plt.show()

def img_show1(img):
    plt.matshow(img, cmap=plt.cm.gray)
    plt.show()

def img_grad_show(img: torch.Tensor) -> None:
    assert img.requires_grad == True
    ygf = img.grad_fn
    print('')
    print('***********cyclic print grads**************')
    while ygf != ():
        print(ygf)
        try:
            if ygf.next_functions[0][0]==None and len(ygf.next_functions)>1:
                ygf = ygf.next_functions[1][0]
            else:
                ygf = ygf.next_functions[0][0]

        except:
            break





