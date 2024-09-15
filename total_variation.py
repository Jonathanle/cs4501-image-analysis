
"""
Total Variation Algorithm - An algorithm designed to iteratively denoise an image using gradient descent.
"""


# Create a method for generating noisy images using gaussian added noise.
# Create an algorithm that iteratively denoises an image using total variation.
# Implement * note for any other details of importance in the how part.

import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    # Convert PIL image to NumPy array
    img_array = np.array(image)
    

    noise = np.random.normal(mean, sigma, img_array.shape)
    
    noisy_img_array = img_array + noise
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    
 
    noisy_img = Image.fromarray(noisy_img_array)
    
    return noisy_img


# TODO: Collet relevant information needed to implement gradient descent image denoising task.
# specification of the image sizes, 
# function to calculatre l2 norm of image relative to images.

class ImageEvaluator: 
    """
    Class that takes in the following initialization parameters
    l - lambda term - adjustable parameter for assigning significance for Data Fidelity Term
    
    """
    def __init__(self, f_image, l = 0.1): 
        self.l = l
        self.f_image = np.array(f_image)

        # When Calculating loss store the most recently used u_image
        self.u_image = None
        self.gradient = None
    # Function for Calculating Metrix for loss
    def forward(self, u_image):
        """
        Given a noisy image "compute" an image and return the result of how good it is.
        """ 

        u_image = np.array(u_image)

        self.u_image = u_image
        return self.l * self.squared_l2_norm(self.f_image, u_image) + self.calculate_sum_gradients(u_image)
    
    def backward(self): 
        """  
        Given the static formula compute the gradient of all the pixels in reference to the image
        # for this kind of function, no chain rule or backpropagation needed.
        """
        # consider how changes in u pixel changes the cost function -
        # what happens to the data fidelity term - cost change to the function? if it is big then change this one alot


        if self.u_image is None: 
            raise Exception("Erroor: u_image loss function not computed, unable to calculate gradien")

        # originall cost function - cost = (u - f)^2 --> gradient = 2 *(u - f)
        data_fidelity_gradient = self.l * 2 * (self.u_image - self.f_image)

        # add the changs that result in the calcuations to the sum_gradients - in particular changing the central differences for surrounding pixels
        # upper? lower? left? right?


        # Employ Strategy of manual computation of the gradients


        # Create the padding and compute around a padded image.

        penalty_gradient = np.zeros(self.u_image.shape)

        grad_x, grad_y = ImageEvaluator.calculate_gradient(self.u_image)




        for i in range(self.u_image.shape[0]):
            for j in range(self.u_image.shape[1]): 
                
                left_gradient, right_gradient, bottom_gradient, top_gradient = (0, 0, 0, 0)

                use_left_gradient, use_right_gradient, use_bottom_gradient, use_top_gradient = (True, True, True, True)

                if i == 0: 
                    use_bottom_gradient = False

                if j == 0: 
                    use_left_gradient = False

                if i == self.u_image.shape[0] - 1: 
                    use_top_gradient = False
                if j == self.u_image.shape[1] - 1:
                    use_right_gradient = False



                if use_left_gradient:
                    # left
                    x_gradient = grad_x[i][j - 1]
                    x_gradient_derivative = 0.5

                    y_gradient = grad_y[i][j - 1]

                    # keep unsimplified to ensure proper computation

            
                    left_gradient = 0.5 * (x_gradient ** 2 + y_gradient **2) ** -0.5 * 2 * x_gradient * x_gradient_derivative


                if use_right_gradient:
                    x_gradient = grad_x[i][j + 1]#(self.u_image[i][j + 2] - self.u_image[i][j]) / 2
                    x_gradient_derivative = -0.5

                    y_gradient = grad_y[i][j + 1]#(self.u_image[i + 1][j + 1] - self.u_image[i - 1][j + 1])

                    # keep unsimplified to ensure proper computation
                    right_gradient = 0.5 * (x_gradient ** 2 + y_gradient **2) ** -0.5 * 2 * x_gradient * x_gradient_derivative

                if use_bottom_gradient:
                    # Bottom 

                    x_gradient = grad_x[i - 1][j] #(self.u_image[i-1][j + 1] - self.u_image[i-1][j - 1]) / 2
                    

                    y_gradient = grad_y[i - 1][j]#(self.u_image[i][j] - self.u_image[i - 2][j]) / 2
                    y_gradient_derivative = 0.5

                    # keep unsimplified to ensure proper computation
                    bottom_gradient = 0.5 * (x_gradient ** 2 + y_gradient **2) ** -0.5 * 2 * y_gradient * y_gradient_derivative

                if use_top_gradient:
                    # Top  (assuming going down is top)

                    x_gradient = grad_x[i + 1][j]#(self.u_image[i+1][j + 1] - self.u_image[i+1][j - 1]) / 2
                    
                    y_gradient = grad_y[i + 1][j]#(self.u_image[i + 2][j] - self.u_image[i][j]) / 2
                    y_gradient_derivative = -0.5

                    # keep unsimplified to ensure proper computation
                    top_gradient = 0.5 * (x_gradient ** 2 + y_gradient **2) ** -0.5 * 2 * y_gradient * y_gradient_derivative

                total_gradient = left_gradient + right_gradient + top_gradient + bottom_gradient

            

                penalty_gradient[i][j] = total_gradient 
        # return the whole sum.

        self.gradient =  data_fidelity_gradient + penalty_gradient

    def step(self):
        
        new_noisy_img = self.u_image - 0.001 * self.gradient 
        return new_noisy_img
    @staticmethod
    def squared_l2_norm(image1, image2):
        """ 
        Given 2 np arrays of the same shape, compute the squared l2 norm

        between the 2 np array
        """

        return np.sum((image1 - image2) ** 2)
    @staticmethod
    def calculate_sum_gradients(u_image):

        grad_x, grad_y = ImageEvaluator.calculate_gradient(u_image)

        magnitude_gradient = (grad_x ** 2 + grad_y ** 2) ** 0.5

        

        return np.sum(magnitude_gradient)

    def calculate_gradient(u_image):
        """ 
        Calculates gradient of an image at (x, y)
        Calculates the central differencea along x and y direction 
        (central === how much the pixel changes going from the center point)
        

        returns - gradient vector for x and y directions 
        """
        # pad the image or calcualting gradients.

        u_image_padded = np.pad(u_image, 1, mode = 'reflect')

        # use a x gradient kernel + y_gradient kernel to calcaulte gradient (Deprecated)
        # Use the strategy of computing the whole array, computing the central gradient

        # problem 1 - calculating the gradient of the pixels
        grad_x = (u_image_padded[1:-1, 2:] - u_image_padded[1:-1, :-2])  / 2 
        grad_y = (u_image_padded[2:, 1:-1] - u_image_padded[:-2, 1:-1]) / 2

        # noticed that the grad attributes remove the padding.
        return grad_x, grad_y
        



if __name__ == '__main__': 
    img = cv2.imread('einstein.jpeg', cv2.IMREAD_GRAYSCALE)

    noisy_img = add_gaussian_noise(img, )

    original_noisy_img = noisy_img

    trainer = ImageEvaluator(noisy_img)

    # similar to neural network, but we are just optimimzing the inputs rather than the parameters that evaluate

    for i in range(0, 1000): 
        print(trainer.forward(noisy_img)) # when calculating the loss function the values seem to go to 0, what is the problem here? 
        trainer.backward()

        noisy_img = trainer.step()
       


    fig, axs = plt.subplots(1, 4, figsize=(18, 6))

    # Display original image
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')


    axs[1].imshow(original_noisy_img, cmap='gray')
    axs[1].set_title('Noisy Image')
    axs[1].axis('off')


    axs[2].imshow(noisy_img, cmap='gray')
    axs[2].set_title('ROF Denoised Image')
    axs[2].axis('off')

    plt.show()