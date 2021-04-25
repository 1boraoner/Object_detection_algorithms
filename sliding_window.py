import matplotlib.image as img_mat
import matplotlib.pyplot as plt
import numpy as np
from cv2 import rectangle
from tensorflow.keras.applications.mobilenet import MobileNet,preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def image_show(original,img):
    
    f,axpr = plt.subplots(2,1)
    axpr[0].imshow(original)
    axpr[1].imshow(img)
    plt.show()
    
    
def sliding_window(img,w_size,step_k):
    
    window_size = w_size
    shape = img.shape
    
    step_size = step_k # pixel step size
    window_images = list()
    for i in range((shape[0] - window_size[0]) // step_size):
        for j in range((shape[1] - window_size[1])//step_size): #shape[1] // window_size[1]

            extracted_window = img[(step_size*i):(step_size*i)+window_size[0],(step_size*j):(step_size*(j) + window_size[1]),:]
            window_images.append(extracted_window)

    return window_images


def draw_box(img,index,windows_images,step_size):

    y_cord = index // ((img.shape[1] - 224)//step_size) 
    x_cord = index % ((img.shape[1] - 224) // (step_size))
    rectangle(img,((x_cord)*step_size,y_cord*step_size) ,((x_cord)*step_size + 224,y_cord*step_size+224),(255, 0, 0), 2)   
    plt.imshow(img)
    plt.show()

def main():
    
    path = "C:\\Users\\user\\Desktop\\python projeler\\Sliding Windows\\sliding_5.jpg"
    img = np.array(img_mat.imread(path))
    img[0] = img[0] / 255
    
    windows_images = sliding_window(img,[224,224],20)

    windows_images_processed = [preprocess_input(np.expand_dims(windows,axis=0)) for windows in windows_images]
    
    model = MobileNet(include_top=True,weights='imagenet')
    
    final_windows_probs = list()
    
    index = 0
    for windows in windows_images_processed:
        
        pred = model.predict(windows)
        if 281<= np.argmax(pred) and np.argmax(pred) <= 285:
            final_windows_probs.append([windows,pred[0,np.argmax(pred)],index])
            print(f"I found the cat!!!, with probs {pred[0,np.argmax(pred)]} imagenet_ind = {np.argmax(pred)}")
        else:
            pass
        
        index += 1
    
    final_windows_probs = np.array(final_windows_probs,dtype=object)

    index_find = final_windows_probs[:,1]                               

    the_most_index = final_windows_probs[np.argmax(index_find)][2]      #window with the most probability

    draw_box(img,the_most_index,windows_images[the_most_index],20)      #draw the box
    
    
    
if __name__ == '__main__':
    main()