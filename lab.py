"""
6.1010 Spring '23 Lab 2: Image Processing 2
"""

#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image
# VARIOUS FILTERS


def get_pixel(image, row, col, boundary_behavior="None"):
    """'gets the pixel at the specific row and column of the image"""
    width = image["width"]
    height = image["height"]

    if 0 <= row < height and 0 <= col < width:
        return image["pixels"][row * width + col]
    else:
        if boundary_behavior == "None":
            return float("inf")
        else:
            if boundary_behavior == "zero":
                return 0
            if boundary_behavior == "extend":
                if row < 0:
                    row = 0
                elif row >= height:
                    row = height - 1
                if col < 0:
                    col = 0
                elif col >= width:
                    col = width - 1
                return image["pixels"][row * width + col]

            if boundary_behavior == "wrap":
                col = col % width
                row = row % height
                return image["pixels"][row * width + col]


def set_pixel(image, row, col, color):
    """changes the specific pixel value at the row and colum specified"""
    width = image["width"]
    image["pixels"][row * width + col] = color


def apply_per_pixel(image, func):
    """Applies a specific function to every
    pixel in the image"""
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [0 for x in range(image["height"] * image["width"])],
    }
    for row in range(image["height"]):
        for col in range(image["width"]):
            color = get_pixel(image, row, col)
            new_color = func(color)
            set_pixel(result, row, col, new_color)
    return result


def inverted(image):
    """Inverts the image by subtracting the pixel from 255"""

    return apply_per_pixel(image, lambda color: 255 - color)


def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """
    pixels = []
    if boundary_behavior in ("zero", "extend", "wrap"):
        for row in range(image["height"]):
            for col in range(image["width"]):
                pixels.append(apply_kernel(image, row, col, kernel, boundary_behavior))
    else:
        return None

    return {"height": image["height"], "width": image["width"], "pixels": pixels}


def apply_kernel(image, row, col, kernel, boundary_behavior):
    """Helper function designed to apply a kernel to the pixels of an image"""
    pixel = 0
    i = 0
    h_kernel = int(math.sqrt(len(kernel)))
    w_kernel = int(math.sqrt(len(kernel)))
    for row_k in range(row - h_kernel // 2, row + h_kernel // 2 + 1):
        for col_k in range(col - w_kernel // 2, col + w_kernel // 2 + 1):

            pixel += get_pixel(image, row_k, col_k, boundary_behavior) * kernel[i]
            i += 1

    return pixel


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    for idx, pix in enumerate(image["pixels"]):
        if pix > 255:
            image["pixels"][idx] = 255
        elif pix < 0:
            image["pixels"][idx] = 0
        else:
            image["pixels"][idx] = round(pix)
    return image


def get_kernel(n):
    """
    Takes in a single element n and returns a kernel that is of size
    n*n in which all values add to 1
    """
    k_value = 1 / (n**2)
    kernel = [k_value] * (n**2)
    return kernel


def edges(image):
    """Takes in an image and applies kernels, Krow and Kcol respectively
    and then gets the square root sum of the squares
     of each pixel from the two images and places the pixel
     on a new image
    """
    new_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [0] * (image["height"] * image["width"]),
    }
    k_row = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    k_col = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    image_r = correlate(image, k_row, boundary_behavior="extend")
    image_c = correlate(image, k_col, boundary_behavior="extend")

    for i in range(len(image["pixels"])):
        color = math.sqrt((image_r["pixels"][i]) ** 2 + (image_c["pixels"][i]) ** 2)
        round(color)
        new_image["pixels"][i] = color
    return round_and_clip_image(new_image)


def splitting(image):
    '''Splits the image into red, green, blue pixels so as 
    to implement filters much more effectively'''
    red_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [],
    }
    blue_image = {"height": image["height"], "width": image["width"], "pixels": []}
    green_image = {"height": image["height"], "width": image["width"], "pixels": []}
    for pix in image["pixels"]:
        red_image["pixels"].append(pix[0])
        green_image["pixels"].append(pix[1])
        blue_image["pixels"].append(pix[2])

    return red_image, green_image, blue_image


def combine_images(red_image, green_image, blue_image):
    colored_pixels = []
    i = 0
    while len(colored_pixels) < red_image["height"] * red_image["width"]:
        colored_pixels.append(
            (red_image["pixels"][i], green_image["pixels"][i], blue_image["pixels"][i])
        )
        i += 1
    return {
        "height": red_image["height"],
        "width": red_image["width"],
        "pixels": colored_pixels,
    }


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def colored_filter(image):
        red_image, green_image, blue_image = splitting(image)
        red_image, green_image, blue_image = (
            filt(red_image),
            filt(green_image),
            filt(blue_image),
        )
        return combine_images(red_image, green_image, blue_image)

    return colored_filter
    # raise NotImplementedError


def make_blur_filter(kernel_size):
    def blur_filter(image):
        new_image = correlate(
            image, get_kernel(kernel_size), boundary_behavior="extend"
        )
        return round_and_clip_image(new_image)

    return blur_filter


def make_sharpen_filter(kernel_size):
    '''Takes a single argument(kernel_size) and returns a function
    that takes in an image as an argument'''
    def sharpen_filter(image):
        pixels = []
        blurred_image = correlate(
            image, get_kernel(kernel_size), boundary_behavior="extend"
        )
        for i in range(len(image["pixels"])):
            pix = 2 * (image["pixels"][i]) - blurred_image["pixels"][i]
            pixels.append(pix)

        new_image = {
            "height": image["height"],
            "width": image["width"],
            "pixels": pixels,
        }
        return round_and_clip_image(new_image)

    return sharpen_filter


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """

    def filter_image(image):
        i = 1
        result = filters[0](image)
        while i < len(filters):
            result = filters[i](result)
            print(result)
            i += 1
        return result

    return filter_image


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    new_image={
        "height":image['height'],
        'width':image['width'],
        'pixels':image['pixels']
    }
    for col in range(ncols):
        grey_color_image = greyscale_image_from_color_image(
            new_image
        )  
        seam = minimum_energy_seam(
            cumulative_energy_map(compute_energy(grey_color_image))
        )
        new_image = image_without_seam(new_image, seam)
    return new_image
    # raise NotImplementedError


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    grey_image = {"height": image["height"], "width": image["width"], "pixels": []}
    for pixel in image["pixels"]:
        pix_val = round(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
        grey_image["pixels"].append(pix_val)
    return grey_image


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    image = {"height": energy["height"], "width": energy["width"], "pixels": []}
    for row in range(energy["height"]):
        for col in range(energy["width"]):
            if row == 0:
                image["pixels"].append(get_pixel(energy, row, col))
            elif col == 0:
                pixel = get_pixel(energy, row, col)
                new_pixel = pixel + min(
                    get_pixel(image, row - 1, col), get_pixel(image, row - 1, col + 1)
                )
                image["pixels"].append(new_pixel)
            elif col == (energy["width"] - 1):
                pixel = get_pixel(energy, row, col)
                new_pixel = pixel + min(
                    get_pixel(image, row - 1, col - 1), get_pixel(image, row - 1, col)
                )
                image["pixels"].append(new_pixel)
            else:
                pixel = get_pixel(energy, row, col)
                new_pixel = pixel + min(
                    get_pixel(image, row - 1, col - 1),
                    get_pixel(image, row - 1, col),
                    get_pixel(image, row - 1, col + 1),
                )
                image["pixels"].append(new_pixel)
    return image


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    indices = []
    # Extract the last row of pixels from the image
    last_row = cem["pixels"][
        (cem["height"] - 1) * cem["width"] : cem["width"]
        + ((cem["height"] - 1) * cem["width"])
    ]
    # Find the index of the pixel with the minimum value in the last row
    index = last_row.index(min(last_row)) + (cem["height"] - 1) * cem["width"]
    # Compute the column index of the pixel with the minimum value in the last row
    ref_col = index - ((cem["height"] - 1) * cem["width"])
    # Loop through the rows of the image from bottom to top
    for row in range(cem["height"] - 1, -1, -1):
        # Compute the index of the current pixel
        pix_index = ref_col + (row * cem["width"])
        # Add the index of the current pixel to the shortest path
        indices.append(pix_index)
        if row > 0:
            above_pixel = get_pixel(cem, row - 1, ref_col)
            left_pixel = get_pixel(cem, row - 1, ref_col - 1)
            right_pixel = get_pixel(cem, row - 1, ref_col + 1)
        min_pix = min(above_pixel, left_pixel, right_pixel)
        if min_pix == above_pixel:
            pass  # No need to update the current column
        elif min_pix == left_pixel:
            ref_col -= 1
        else:
            ref_col += 1

    return indices


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    new_image = {
        "height": image["height"],
        "width": image["width"] - 1,
        "pixels": [],
    }
    for index, pix in enumerate(image["pixels"]):
        if index not in seam:
            new_image["pixels"].append(pix)
    return new_image
    # raise NotImplementedError


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    def case1():
        image = load_color_image("test_images/twocats.png")
        new_image = seam_carving(image, 100)
        save_color_image(new_image, "test_images/modifiedtwocats.png", "PNG")

    case1()
