# Import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
#import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
#import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
#import other dependencies  
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

#Build app and layout
class CamApp(App):

    #define the base method
    def build(self):
        #Main layout components
        self.web_cam=Image(size_hint=(1,.8))
        self.button=Button(text="Verify",on_press=self.verify, size_hint=(1,.1))
        self.verification_label=Label(text="Verification Uninitiated",size_hint=(1,.1))

        #add items to layout
        layout=BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        
        #load tensorflow keras model
        self.model=tf.keras.models.load_model('siamesemodelV2.h5', custom_objects={'L1Dist':  L1Dist})


        #Setup video capture device
        self.capture=cv2.VideoCapture(0)
        #run the update function (run the update function every 33 times every second )
        Clock.schedule_interval(self.update,1.0/33.0)
        return layout
    
    #Run continuously to get the webcam feed
    def update(self,*args):
        #read frame from openCv
        ret,frame = self.capture.read()
        frame=frame[120:120+250,200:200+250,:]
        
        #update the image by convert image to texture
        #flip the image horizontally
        buf=cv2.flip(frame, 0).tostring()
        #convert the image to image texture
        #fram.shape[1]:height,frame.shape[0]:weight
        img_texture=Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        #convert the opencv buffer to texture
        img_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        self.web_cam.texture=img_texture


    #load image from file and convert it to 105*105 pixels per
    def preprocess(self,file_path):
        #read in image from file path
        byte_img=tf.io.read_file(file_path)
        #load in the image and decode it
        img=tf.image.decode_jpeg(byte_img)
        #preprocessing steps
        img=tf.image.resize(img,(105,105))#resize the img to 105*105*3
        img=img/255.0 #scale img
        return img  
    
    #verification function to verify person
    def verify(self,*args):
        #specify thresholds
        detection_threshold=0.5
        verification_threshold=0.8
        
        #save the current image to file
        SAVE_PATH=os.path.join('application_data','input_image','input_image.jpg')
        ret,frame = self.capture.read()
        frame=frame[120:120+250,200:200+250,:]
        cv2.imwrite(SAVE_PATH,frame)


        #build a result array
        results=[]    
        #looping through every single image inside the verification image folder
        for image in os.listdir(os.path.join('application_data','verification_images')):
            #grab an input img from the webcam and store it in the input folder and call it input_image.jpg        
            input_img = self.preprocess(SAVE_PATH)
            validation_img = self.preprocess(os.path.join('application_data','verification_images',image))
            #make pred
            #wrapping data into list (encapsuler data) because we've only got a single sample(1 seul echantillon)
            result = self.model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
            results.append(result)
            
            
        #detection threshold: Metrics above which a prediction is considered positive (matches)
        
        #how many positive prediction are actually passing the detection thres
        detection = np.sum(np.array(results)> detection_threshold) 
            
            
        # verification threshold:the proportion of positve pred /total positive samples
        
        verification=detection / len(os.listdir(os.path.join('application_data','verification_images')))
        verified=verification > verification_threshold #this return true or false
            
        #set verification test
        self.verification_label.text ='verified' if verified==True else 'Unverified'

        #Log out details
        Logger.info(results)
        Logger.info(detection)#how many exemples pass the threshold
        Logger.info(verification)#pourcentage de verified
        Logger.info(verified)

        return results,verified

    
if __name__ == "__main__":
    CamApp().run()