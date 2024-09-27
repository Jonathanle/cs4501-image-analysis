
"""
Total Variation Algorithm - An algorithm designed to iteratively denoise an image using gradient descent.
"""


# Create a method for generating noisy images using gaussian added noise.
# Create an algorithm that iteratively denoises an image using total variation.
# Implement * note for any other details of importance in the how part.
# DF: Investigate what Problems in bad gradient calculations may have when calcualting gradients, and if the gradients are valid 

import numpy as np
from PIL import Image
import cv2
from itertools import product
import pdb
        

import matplotlib.pyplot as plt
from matplotlib import gridspec

def add_gaussian_noise_normalize(image, mean=0, sigma=0.1):
    """ 
    Add random gaussian noise and normalize image values frmo cv2
    """



    #img_array = np.array(image)

    image = image.astype(float) / 255.0

    
    

    noise = np.random.normal(mean, sigma, image.shape)
    
    noisy_img_array = image + noise
    noisy_img_array = np.clip(noisy_img_array, 0, 1).astype(np.float64)

    
    
 
    #noisy_img = Image.fromarray(noisy_img_array)
    
    return noisy_img_array


# TODO: Collet relevant information needed to implement gradient descent image denoising task.
# specification of the image sizes, 
# function to calculatre l2 norm of image relative to images.

class ImageEvaluator: 
    """
    Class that takes in the following initialization parameters
    l - lambda term - adjustable parameter for assigning significance for Data Fidelity Term
    
    """
    def __init__(self, f_image, l = 5, learning_rate = 0.01): 
        self.l = l
        self.learning_rate = learning_rate
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

                    #if x_gradient + y_gradient == 0: 
                        #pdb.set_trace()

                
                    left_gradient = 0.5 * (x_gradient ** 2 + y_gradient **2 + 1e-8) ** -0.5 * 2 * x_gradient * x_gradient_derivative


                if use_right_gradient:
                    x_gradient = grad_x[i][j + 1]#(self.u_image[i][j + 2] - self.u_image[i][j]) / 2
                    x_gradient_derivative = -0.5

                    y_gradient = grad_y[i][j + 1]#(self.u_image[i + 1][j + 1] - self.u_image[i - 1][j + 1])

                    # keep unsimplified to ensure proper computation
                    right_gradient = 0.5 * (x_gradient ** 2 + y_gradient **2 + 1e-8) ** -0.5 * 2 * x_gradient * x_gradient_derivative

                if use_bottom_gradient:
                    # Bottom 

                    x_gradient = grad_x[i - 1][j] #(self.u_image[i-1][j + 1] - self.u_image[i-1][j - 1]) / 2
                    

                    y_gradient = grad_y[i - 1][j]#(self.u_image[i][j] - self.u_image[i - 2][j]) / 2
                    y_gradient_derivative = 0.5

                    # keep unsimplified to ensure proper computation
                    bottom_gradient = 0.5 * (x_gradient ** 2 + y_gradient **2 + 1e-8) ** -0.5 * 2 * y_gradient * y_gradient_derivative

                if use_top_gradient:
                    # Top  (assuming going down is top)

                    x_gradient = grad_x[i + 1][j]#(self.u_image[i+1][j + 1] - self.u_image[i+1][j - 1]) / 2
                    
                    y_gradient = grad_y[i + 1][j]#(self.u_image[i + 2][j] - self.u_image[i][j]) / 2
                    y_gradient_derivative = -0.5

                    # keep unsimplified to ensure proper computation
                    top_gradient = 0.5 * (x_gradient ** 2 + y_gradient **2 + 1e-8) ** -0.5 * 2 * y_gradient * y_gradient_derivative

                total_gradient = left_gradient + right_gradient + top_gradient + bottom_gradient

            

                penalty_gradient[i][j] = total_gradient 
        # return the whole sum.

        self.gradient =  data_fidelity_gradient + penalty_gradient


        #pdb.set_trace()
    def step(self):
        
        new_noisy_img = self.u_image - self.learning_rate * self.gradient 
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

        
        #print(grad_x)    


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
    img = cv2.imread('Einstein.jpeg', cv2.IMREAD_GRAYSCALE)

    img = np.array(img)
    img_original = add_gaussian_noise_normalize(img, sigma = 0)

    variances = [0.1, 0.01, 0.001]
    lambdas = [1, 2, 5, 10]
    learning_rates = [0.01]
    epochs = [25]

    
    fig = plt.figure(figsize=(26, 10))


    n_cols = 12
    # Create GridSpec
    gs = gridspec.GridSpec(2, 12 + 1, height_ratios=[1, 1], width_ratios=[1.5] + [1]*(13-1))

    # Adjust spacing
    gs.update(hspace=0.01, wspace=0.01)
    
    
    
    
    # Display original image
    ax_orig_top = fig.add_subplot(gs[0, 0])
    ax_orig_top.imshow(img_original, cmap='gray')
    ax_orig_top.set_title('Original Image')
    ax_orig_top.axis('off')

    ax_orig_bottom = fig.add_subplot(gs[1, 0])
    ax_orig_bottom.imshow(img_original, cmap='gray')
    ax_orig_bottom.set_title('Original Image')
    ax_orig_bottom.axis('off')


    # Define a figure to plot convergence graph

    #fig, ax = plt.subplot(1,1)
    

    # Process and display other images
    for i, (epoch, learning_rate, lambda_, variance) in enumerate(product(epochs, learning_rates, lambdas, variances), start=1):
        print(f"iteration {i}")
        print((epoch, learning_rate, lambda_, variance))
        noisy_img = add_gaussian_noise_normalize(img, sigma=variance)
        original_noisy_img = noisy_img

        trainer = ImageEvaluator(noisy_img, l=lambda_, learning_rate=learning_rate)


        convergence = []

        for j in range(0, epoch):
            convergence.append(trainer.forward(noisy_img))
            trainer.backward()
            noisy_img = trainer.step()

        

        ax_top = fig.add_subplot(gs[0, i])
        ax_top.imshow(original_noisy_img.copy(), cmap='gray', vmin=0, vmax=1)
        ax_top.set_title('O')
        ax_top.axis('off')

        ax_bottom = fig.add_subplot(gs[1, i])
        ax_bottom.imshow(noisy_img.copy(), cmap='gray', vmin=0, vmax=1)
        ax_bottom.set_title(f'DN\n lr - {learning_rate} \nlambda - {lambda_} \n-variance -{variance}')
        ax_bottom.axis('off')

        """
        plt.figure(figsize=(10, 6))
        plt.plot(list(range(epoch)), convergence, linestyle = '-', marker='o')
        plt.title(f'E[u] vs Epochs ($\lambda = {lambda_}, lr = 0.01,\sigma = {variance}$)')
        plt.xlabel('Epochs')
        plt.ylabel('E[u]')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        """

    plt.tight_layout()
    plt.show()