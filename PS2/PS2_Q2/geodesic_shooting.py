import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy import ndimage

import pdb



# ideas np arrays arr[1,1] get the hwole array slice np arr[1][2] go into the slices.

def calculate_divergence(v_t):
    """
    Given a velocity field (2, height, width)- calculate scalar divergence at each point
    """
    v_y = v_t[0]
    v_x = v_t[1]

    # calculate partial derivatives of the velocity fucntion
    div_x = central_difference_x(v_x)
    div_y = central_difference_y(v_y)


    return div_x + div_y
    


def forward_difference_x(image):
    rows, cols = image.shape
    d = np.zeros((rows,cols))
    d[:,1:cols-1] = image[:,1:cols-1] - image[:,0:cols-2];
    d[:,0] = image[:,0] - image[:,cols-1];
    return d



"""
Given helper functions for calculating the gradient
"""
def forward_difference_y(image):
    rows, cols = image.shape
    d = np.zeros((rows,cols))
    d[1:rows-1,:] = image[1:rows-1,:] - image[0:rows-2,:];
    d[0,:] = image[0,:] - image[rows-1,:];
    return d

def backward_difference_x(image):
    rows, cols = image.shape
    d = np.zeros((rows,cols))
    d[:,1:cols-1] = image[:,1:cols-1] - image[:,2:cols]
    d[:,-1] = image[:,-1] - image[:,0]
    return d

def backward_difference_y(image):
    rows, cols = image.shape
    d = np.zeros((rows,cols))
    d[1:rows-1,:] = image[1:rows-1,:] - image[2:rows,:]
    d[-1,:] = image[-1,:] - image[0,:]
    return d

def central_difference_x(image):
    cdif=(forward_difference_x(image)+backward_difference_x(image))/2
    return cdif

def central_difference_y(image):
    cdif=(forward_difference_y(image)+backward_difference_y(image))/2
    return cdif


def calculate_jaccobian(v_t):
    """

    Given a velocity field v_t of shape (2, height, width)
    
    , return a (height, width, 2,2,) np array containing the jaccobi matrix for every pixel

    """


    v_y = v_t[0] # in this case I provide interface for acccessing np array by specifiying which direction gradient i want.
    v_x = v_t[1]

    # Compute gradients
    vxx, vxy = central_difference_x(v_x), central_difference_y(v_x)
    vyx, vyy = central_difference_x(v_y), central_difference_y(v_y)
    
    # Stack into Jacobian tensor
    jacobian = np.stack([
        np.stack([vxx, vxy], axis=-1),
        np.stack([vyx, vyy], axis=-1)
    ], axis=-2) # When I access the information, what kind of interface do I want to specificy when requesting for specific objects?
    # axis -1, when I am accessing an element, i want to request height, width then ask for either vxx or vyy [0] [1] - when i stack. 
    # then when I stack each of these (height, width, 2) arrays, the property that I am stacking by is by which gradient i differentiate,
    # i want this dimension to be accessed through the second to last index - "library interface organizing objective"




    #latest_books = np.max(library, axis=2) --> in this case what does axis mean? - It will divide into 2 groups, then return
    # the group contains an array - that one can index by to get the maximum value, based on those factors, returns the group index 
    # along that axis 
    # idea - using max to analyze shapes along a specific axis - goal - for each position where we keep all factors constant, find the
    # maximum value along this axis - Analysis Objective

    # Query Objective - get a specific group.

    
    return jacobian


def change_velocity_field(v_t): 
    """ 
    Accepts v_t as an np array of shape (2, height, width), returns a dv / dt 
    """

    # Rearrange the axes so that the dimension is added towards the end

    
    # Move the first axis to the back and add a dimension for matrix multiplication (2x1 dimension)
    
    
    # Create jaccobian matrix of shape (height, width, 2,2) 
    jaccobian = calculate_jaccobian(v_t)

    # Create a new Object for multiplication (height, width, 2,1)
    v_t_standardized = np.moveaxis(v_t, 0, -1)
    v_t_standardized = np.expand_dims(v_t_standardized, axis = -1)


    jaccobian_transposed = np.transpose(jaccobian, axes = (0, 1, 3, 2))

    assert v_t_standardized.shape[2] == 2 and v_t_standardized.shape[3] == 1
    assert jaccobian_transposed.shape[2] == 2 and jaccobian_transposed.shape[3] == 2



    # Perform matrix multiplication
    result_matmul = np.matmul(jaccobian_transposed, v_t_standardized)
    result_matmul2 = np.matmul(jaccobian, v_t_standardized)


    # Calculate divergence * vectors
    div_scalars = calculate_divergence(v_t)  
    div_operator_value =  v_t_standardized * div_scalars[:, :, np.newaxis, np.newaxis]

    # broadcasting process Can we infer multiplicaiton of diff sizes? first match size then interpolate ?
    # - promote the shapes --> fill in with values - 
    # how broadcasting works? - idea 2x1 vector allows to be multiplied by 1x1 vector, 
    # what abot in other shapes? like (2,) (1,)? (2,1) (1,)? (2,) (1, 1) --> larger vector is assumed to be multiplied.

    sum = result_matmul + result_matmul2 + div_operator_value



    # Gaussian Filter y + x components
    smoothed_vector_field = np.zeros_like(sum)
    sigma = 1 # Define the standard deviation of the Gaussian kernel

    # Apply Gaussian filter separately to each vector component
    for i in range(2):
        smoothed_vector_field[:, :, i, 0] = ndimage.gaussian_filter(sum[:, :, i, 0], sigma=sigma)


    smoothed_vector_field *= -1

    # transform the array back into standard v_t format.


    # this returns the change in velocity over a change in time not the actual velocity

    v_t_final = np.squeeze(smoothed_vector_field , axis = -1)
    v_t_final = np.moveaxis(v_t_final, -1, 0) # the indice indicates the final position that "pushes" out other elements.

    # Intermediate value - return the jaccobian value

    return v_t_final

def apply_velocity_field(image, v_t, dt=0.1):
    """
    Given a v_t velocity field and image, transform the image with the velocity field
    + interpolate the transformed coordinates.
    """
    # Create coordinate grids
    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w]
    
    # Apply velocity field to coordinates (change the coordinates)
    new_y = y + v_t[0] * dt
    new_x = x + v_t[1] * dt
    
    # Flatten coordinates
    #points = np.stack([y.flatten(), x.flatten()], axis=-1)
    #new_points = np.stack([new_y.flatten(), new_x.flatten()], axis=-1)
    coords = np.stack([new_y, new_x])

    # why is flatten importnat? idk why I woudl need this here? how does axis relate?
    print(coords.shape)


    # Map the images to new Coordinates and Interpolate

    deformed_image = ndimage.map_coordinates(image, coords, order=3, mode='nearest')
    
    # Reshape back to image dimensions
    deformed_image = deformed_image.reshape(h, w)
    
    return deformed_image

# Example usage
# Assume we have a 100x100 image and a corresponding velocity field


'''Read in data as a 2, 100, 100 vector field'''
v_t= torch.load('code+data/v0.pt').numpy()
print("Dimension of velocity V0: ", v_t.shape)

print(v_t)

# Read in data as a 1, 100, 100  image
image= torch.load('code+data/source.pt').numpy()
print("Dimension of image: ", image.shape)





def generate_random_velocity_field(image_shape, smoothing_variance):
    # Generate random fields from normal distribution
    random_field = np.random.normal(size=(2, *image_shape))
    
    # Smooth the random fields
    smoothed_field = ndimage.gaussian_filter(random_field, sigma=(0, np.sqrt(smoothing_variance), np.sqrt(smoothing_variance)))
    
    return smoothed_field

def compute_image_gradient(image):
    grad_y, grad_x = np.gradient(image)
    return np.stack([grad_x, grad_y])

def create_initial_velocity(source_image, smoothing_variance):
    # Generate smoothed random fields
    random_field = generate_random_velocity_field(source_image.shape, smoothing_variance)
    
    # Compute image gradient
    grad = compute_image_gradient(source_image)
    
    # Compute initial velocity field
    v0 = random_field * grad
    
    return v0


"""
# Example usage
image_shape = (100, 100)  # Replace with your actual image shape
smoothing_variances = [2.0, 4.0, 8.0]

for variance in smoothing_variances:
    v0 = create_initial_velocity(source_image, variance)
    print(f"Initial velocity field generated with smoothing variance {variance}")
    print(f"Shape of v0: {v0.shape}")  # Should be (2, height, width)
    # You can now use v0[0] as v0_x and v0[1] as v0_y

"""






step_size = 0.1
image_final = image.copy()
for i in range(11):


    image_final = apply_velocity_field(image_final, v_t) 



    dv_dt = change_velocity_field(v_t)
    v_t += step_size * dv_dt

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(image_final, cmap='gray')
    ax2.set_title('Deformed Image')
    plt.show()
    """


    

# Get the coordinate of the height and width, the velocity vectro, why is this incorrect?
#print(v_t[:].shape) --> get the array
#print(v_t[:][1].shape) # Here I really just asked let's get the v_t slice, then get the first slice - here im still indexing by first index
#print(v_t[:][1][1].shape) # I asked here that given 


#print(v_t[:, 1, 1].shape) # query a subarray with the specific dimensions.
#print(jacobian[1][1])

# Apply the velocity field
#deformed_image = apply_velocity_field(image, v_t)
#v_t = change_velocity_field(v_t)

assert v_t.shape == (2, 100, 100)
assert image.shape == (100, 100)

print(image_final)


# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(image_final, cmap='gray')
ax2.set_title('Deformed Image')
plt.show()

print()