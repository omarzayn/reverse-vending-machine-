import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2 
import RPi.GPIO as GPIO
import time


def ultrasonic () :
    GPIO.setmode(GPIO.BCM)

    TRIG = 23
    ECHO = 24

    print ("Distance Measurement In Progress")

    GPIO.setup(TRIG,GPIO.OUT)
    GPIO.setup(ECHO,GPIO.IN)
    try:
        while True:
        
            GPIO.output(TRIG, False)
            print ("Waiting For Sensor To Settle")
            time.sleep(2)
        
            GPIO.output(TRIG, True)
            time.sleep(0.00001)
            GPIO.output(TRIG, False)
        
            while GPIO.input(ECHO)==0:
                pulse_start = time.time()
        
            while GPIO.input(ECHO)==1:
                pulse_end = time.time()
        
            pulse_duration = pulse_end - pulse_start
        
            distance = pulse_duration * 17150
        
            distance = round(distance, 2)
        
            print ("Distance: ",distance,"cm")
            
            if distance < 15.5 :
                return 1
        
    except KeyboardInterrupt:
    
        # If there is a KeyboardInterrupt (when you press ctrl+c), exit the program
        print("Cleaning up!")
        GPIO.cleanup()
        
        
#-----------------------------------------------------------------------------        

       
def classification () :
    video=cv2.VideoCapture(0)


    check, frame=video.read()


    cv2.imshow("asm al sora", frame)


    cv2.imwrite("test_photo.jpg", frame)



    cv2.waitKey(3000)




    video.release()

    cv2.destroyAllWindows()	


    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open('test_photo.jpg')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    output = list(prediction[0])
    certainity_index = output.index(max(output))

    label = ["Plastic","Metal","Glass","cardboard"]

     #print("\n \n  " ,label[certainity_index])
    
    return label[certainity_index]

    #print(certainity_index)

#----------------------------------------------------------------------------
    


check_distance = ultrasonic()


classify_object = classification()

print("we have found {} object !".format(classify_object))






    
    
    
    
    
       