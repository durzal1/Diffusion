# About 
This custom diffusion model was trained from scratch on a gtx 2080 Super in ~ 10-15 hours over 27 epochs. 

The problem this is solving is creating AI-generated "hand drawn" digits. 

# Architecture 

The model starts off with an image of pure normally distributed noise.

Then the model iteratively subtracts noise until only a pure image remains without noise. 

A U-Net is used to predict the amount of noise to remove from the image. 

# Results
This distribution of images showcases the evolution of the model.

![image](https://github.com/user-attachments/assets/0120fc78-7f29-47d5-9b6e-1be21a1710aa)
![image](https://github.com/user-attachments/assets/30fbcefa-8677-47e2-beb7-03a8ddadf481)
![image](https://github.com/user-attachments/assets/88935aae-4127-4c3a-96ef-86683b619dc3)
![image](https://github.com/user-attachments/assets/14106fc6-bb01-43a5-9270-94a1d749e0c5)
![image](https://github.com/user-attachments/assets/aebad7e3-e1ed-4bee-9e1e-6ed9228de699)

