from skimage import filters


def enhance_image(photo): #From jpg pictures to black and white vectors
    photo_new = []

    val= filters.threshold_otsu(photo)
    new_photo = get_new_photo(photo, val)
    make_letter_black(new_photo)
    return new_photo

def enhance_image_neural_net(photo):
    new_photo = []
    for i in range(len(photo)):
        new_photo.append([])
        for j in range(len(photo[i])):
            new_photo[i].append(photo[i][j]/255)
    return new_photo


def get_new_photo(photo, val):
    image = []
    counter = 0
    for row in photo:
        image.append([])
        for pixel in row:
            if(pixel < val):
                image[counter].append(0)
            else:
                image[counter].append(255)
        counter += 1
    return image


def make_letter_black(photo):
    border_sum = 0
    pixel_counter = 0
    for i in range(len(photo)):
        if i == 0 or i == len(photo)-1:
            for j in range(len(photo[i])):
                border_sum += photo[i][j]
                pixel_counter += 1
        else:
            border_sum += photo[i][-1]
            border_sum += photo[i][0]
            pixel_counter += 2
    border_average = border_sum/pixel_counter

    if 255-border_average > 127.5:
        for i in range(len(photo)):
            for j in range(len(photo[i])):
                if photo[i][j] == 255:
                    photo[i][j] = 0
                else:
                    photo[i][j] = 255

def make_letter_black_neural_net(photo):
    border_sum = 0
    pixel_counter = 0
    for i in range(len(photo)):
        if i == 0 or i == len(photo)-1:
            for j in range(len(photo[i])):
                border_sum += photo[i][j]
                pixel_counter += 1
        else:
            border_sum += photo[i][-1]
            border_sum += photo[i][0]
            pixel_counter += 2
    border_average = border_sum/pixel_counter

    if 1-border_average > 0.5:
        for i in range(len(photo)):
            for j in range(len(photo[i])):
                photo[i][j] = 1 - photo[i][j]

